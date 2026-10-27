//! FLUX.2 text-to-image models from Black Forest Labs.
//!
//! FLUX.2 is a rectified-flow transformer. Compared with FLUX.1, the text conditioning comes from
//! the hidden states of three intermediate layers of a language model (Mistral 3 for FLUX.2
//! [dev], Qwen3 for FLUX.2 [klein]), the modulation is shared by all the blocks of a stream, the
//! MLPs are SwiGLU, RoPE runs over four position axes, and the VAE has 32 latent channels, packed
//! 2x2 into 128-channel tokens and normalized by batch-norm statistics.
//!
//! This module follows the diffusers layout of the checkpoints:
//! - [`model::Flux2`], the transformer (`transformer/`),
//! - [`Vae`], the autoencoder and its latent statistics (`vae/`),
//! - [`Scheduler`], the flow-matching Euler schedule (`scheduler/`),
//! - helpers to build the text embeddings, position ids and packed latents.
//!
//! - 💻 [GitHub](https://github.com/black-forest-labs/flux2)
//! - 🤗 [FLUX.2 [dev]](https://huggingface.co/black-forest-labs/FLUX.2-dev)

use candle::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use crate::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig};

pub mod model;

/// Stacks the hidden states of the text-encoder layers into `(batch, seq, layers * hidden)`
/// embeddings, interleaving them per token as the reference does.
pub fn stack_text_embeddings(hidden_states: &[Tensor]) -> Result<Tensor> {
    let (b, l, _) = hidden_states[0].dims3()?;
    Tensor::stack(hidden_states, 2)?.reshape((b, l, ()))
}

/// The schedule shift of the reference implementation, fitted to the image token count and the
/// number of steps.
pub fn empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
    let (a1, b1) = (8.73809524e-05, 1.89833333);
    let (a2, b2) = (0.00016927, 0.45666666);
    let seq = image_seq_len as f64;
    if image_seq_len > 4300 {
        return a2 * seq + b2;
    }
    let m_200 = a2 * seq + b2;
    let m_10 = a1 * seq + b1;
    let a = (m_200 - m_10) / 190.;
    let b = m_200 - 200. * a;
    a * num_steps as f64 + b
}

fn default_train_timesteps() -> usize {
    1000
}

fn default_shift() -> f64 {
    1.
}

fn default_time_shift_type() -> String {
    "exponential".to_string()
}

/// The flow-matching Euler scheduler config, `scheduler/scheduler_config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SchedulerConfig {
    #[serde(default = "default_train_timesteps")]
    pub num_train_timesteps: usize,
    #[serde(default = "default_shift")]
    pub shift: f64,
    #[serde(default)]
    pub use_dynamic_shifting: bool,
    #[serde(default = "default_time_shift_type")]
    pub time_shift_type: String,
    #[serde(default)]
    pub shift_terminal: Option<f64>,
    #[serde(default)]
    pub invert_sigmas: bool,
    #[serde(default)]
    pub stochastic_sampling: bool,
    #[serde(default)]
    pub use_karras_sigmas: bool,
    #[serde(default)]
    pub use_exponential_sigmas: bool,
    #[serde(default)]
    pub use_beta_sigmas: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 1.,
            use_dynamic_shifting: true,
            time_shift_type: "exponential".to_string(),
            shift_terminal: None,
            invert_sigmas: false,
            stochastic_sampling: false,
            use_karras_sigmas: false,
            use_exponential_sigmas: false,
            use_beta_sigmas: false,
        }
    }
}

/// The flow-matching Euler scheduler of the FLUX.2 pipelines.
#[derive(Debug, Clone)]
pub struct Scheduler {
    /// `num_steps + 1` decreasing noise levels, ending at 0.
    pub sigmas: Vec<f64>,
}

impl Scheduler {
    /// The schedule for `num_steps` steps over `image_seq_len` latent tokens.
    pub fn new(cfg: &SchedulerConfig, num_steps: usize, image_seq_len: usize) -> Result<Self> {
        let unsupported = [
            (cfg.invert_sigmas, "invert_sigmas"),
            (cfg.stochastic_sampling, "stochastic_sampling"),
            (cfg.use_karras_sigmas, "use_karras_sigmas"),
            (cfg.use_exponential_sigmas, "use_exponential_sigmas"),
            (cfg.use_beta_sigmas, "use_beta_sigmas"),
        ];
        for (set, name) in unsupported {
            if set {
                candle::bail!("flux2 scheduler: {name} is not supported")
            }
        }
        if num_steps == 0 {
            candle::bail!("flux2 scheduler: at least one step is needed")
        }
        // sigmas = linspace(1, 1 / n, n), stored in f32 as in the reference.
        let n = num_steps as f64;
        let mut sigmas: Vec<f64> = (0..num_steps)
            .map(|i| {
                let s = if num_steps == 1 {
                    1.
                } else {
                    1. + (1. / n - 1.) * i as f64 / (n - 1.)
                };
                s as f32 as f64
            })
            .collect();
        if cfg.use_dynamic_shifting {
            let mu = empirical_mu(image_seq_len, num_steps);
            for s in sigmas.iter_mut() {
                *s = match cfg.time_shift_type.as_str() {
                    "exponential" => mu.exp() / (mu.exp() + (1. / *s - 1.)),
                    "linear" => mu / (mu + (1. / *s - 1.)),
                    other => candle::bail!("flux2 scheduler: unknown time_shift_type {other}"),
                };
            }
        } else {
            for s in sigmas.iter_mut() {
                *s = fixed_shift(*s, cfg.shift);
            }
        }
        if let Some(terminal) = cfg.shift_terminal {
            let last = 1. - sigmas[num_steps - 1];
            let scale = last / (1. - terminal);
            for s in sigmas.iter_mut() {
                *s = 1. - (1. - *s) / scale;
            }
        }
        let mut sigmas: Vec<f64> = sigmas.into_iter().map(|s| s as f32 as f64).collect();
        sigmas.push(0.);
        Ok(Self { sigmas })
    }

    /// The number of denoising steps.
    pub fn len(&self) -> usize {
        self.sigmas.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One Euler step from `sigmas[step]` to `sigmas[step + 1]`, computed in f32.
    pub fn step(&self, latents: &Tensor, velocity: &Tensor, step: usize) -> Result<Tensor> {
        let dt = self.sigmas[step + 1] - self.sigmas[step];
        let dtype = latents.dtype();
        (latents.to_dtype(DType::F32)? + (velocity.to_dtype(DType::F32)? * dt)?)?.to_dtype(dtype)
    }
}

/// The resolution-independent schedule shift, `shift * s / (1 + (shift - 1) * s)`.
fn fixed_shift(s: f64, shift: f64) -> f64 {
    shift * s / (1. + (shift - 1.) * s)
}

/// `(batch, channels, height, width)` -> `(batch, height * width, channels)`.
pub fn pack(latents: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = latents.dims4()?;
    latents
        .reshape((b, c, h * w))?
        .transpose(1, 2)?
        .contiguous()
}

/// `(batch, height * width, channels)` -> `(batch, channels, height, width)`.
pub fn unpack(latents: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let (b, _, c) = latents.dims3()?;
    latents.transpose(1, 2)?.reshape((b, c, height, width))
}

/// Folds each 2x2 patch of `(batch, c, 2h, 2w)` latents into channels: `(batch, 4c, h, w)`.
pub fn patchify(latents: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = latents.dims4()?;
    latents
        .reshape((b, c, h / 2, 2, w / 2, 2))?
        .permute((0, 1, 3, 5, 2, 4))?
        .reshape((b, c * 4, h / 2, w / 2))
}

/// The inverse of [`patchify`].
pub fn unpatchify(latents: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = latents.dims4()?;
    latents
        .reshape((b, c / 4, 2, 2, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((b, c / 4, h * 2, w * 2))
}

fn default_layers_per_block() -> usize {
    2
}

fn default_latent_channels() -> usize {
    32
}

fn default_norm_groups() -> usize {
    32
}

fn default_block_out_channels() -> Vec<usize> {
    vec![128, 256, 512, 512]
}

fn default_bn_eps() -> f64 {
    1e-4
}

/// The VAE config, `vae/config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VaeConfig {
    #[serde(default = "default_block_out_channels")]
    pub block_out_channels: Vec<usize>,
    #[serde(default)]
    pub decoder_block_out_channels: Option<Vec<usize>>,
    #[serde(default = "default_layers_per_block")]
    pub layers_per_block: usize,
    #[serde(default = "default_latent_channels")]
    pub latent_channels: usize,
    #[serde(default = "default_norm_groups")]
    pub norm_num_groups: usize,
    #[serde(default = "default_true")]
    pub use_quant_conv: bool,
    #[serde(default = "default_true")]
    pub use_post_quant_conv: bool,
    #[serde(default = "default_true")]
    pub mid_block_add_attention: bool,
    #[serde(default = "default_bn_eps")]
    pub batch_norm_eps: f64,
}

fn default_true() -> bool {
    true
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            block_out_channels: default_block_out_channels(),
            decoder_block_out_channels: None,
            layers_per_block: 2,
            latent_channels: 32,
            norm_num_groups: 32,
            use_quant_conv: true,
            use_post_quant_conv: true,
            mid_block_add_attention: true,
            batch_norm_eps: 1e-4,
        }
    }
}

/// The FLUX.2 VAE: a diffusers `AutoencoderKL` whose patchified latents are normalized with the
/// running statistics of a batch-norm layer.
#[derive(Debug)]
pub struct Vae {
    vae: AutoEncoderKL,
    /// `(1, 4 * latent_channels, 1, 1)`.
    bn_mean: Tensor,
    bn_std: Tensor,
}

impl Vae {
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        if cfg
            .decoder_block_out_channels
            .as_ref()
            .is_some_and(|d| *d != cfg.block_out_channels)
        {
            candle::bail!("flux2 vae: separate decoder channels are not supported")
        }
        if !cfg.mid_block_add_attention {
            candle::bail!("flux2 vae: a mid block without attention is not supported")
        }
        let vae_cfg = AutoEncoderKLConfig {
            block_out_channels: cfg.block_out_channels.clone(),
            layers_per_block: cfg.layers_per_block,
            latent_channels: cfg.latent_channels,
            norm_num_groups: cfg.norm_num_groups,
            use_quant_conv: cfg.use_quant_conv,
            use_post_quant_conv: cfg.use_post_quant_conv,
        };
        let vae = AutoEncoderKL::new(vb.clone(), 3, 3, vae_cfg)?;
        let c = 4 * cfg.latent_channels;
        let shape = (1, c, 1, 1);
        let bn_mean = vb.get(c, "bn.running_mean")?.reshape(shape)?;
        let bn_var = vb.get(c, "bn.running_var")?.reshape(shape)?;
        let bn_std = (bn_var + cfg.batch_norm_eps)?.sqrt()?;
        Ok(Self {
            vae,
            bn_mean,
            bn_std,
        })
    }

    /// Decodes normalized, patchified `(batch, 4 * latent_channels, h, w)` latents to
    /// `(batch, 3, 16 * h, 16 * w)` images in `[-1, 1]`.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let latents = latents
            .broadcast_mul(&self.bn_std)?
            .broadcast_add(&self.bn_mean)?;
        self.vae.decode(&unpatchify(&latents)?)
    }

    /// Encodes `(batch, 3, H, W)` images in `[-1, 1]` to normalized, patchified latents, using
    /// the mode of the latent distribution.
    pub fn encode(&self, images: &Tensor) -> Result<Tensor> {
        let dist = self.vae.encode(images)?;
        let latents = patchify(dist.mode())?;
        latents
            .broadcast_sub(&self.bn_mean)?
            .broadcast_div(&self.bn_std)
    }
}

/// Classifier-free guidance: `uncond + scale * (cond - uncond)`.
pub fn apply_cfg(cond: &Tensor, uncond: &Tensor, scale: f64) -> Result<Tensor> {
    uncond + ((cond - uncond)? * scale)?
}

/// Converts `(batch, 3, H, W)` decoded images in `[-1, 1]` to `u8`.
pub fn to_u8_image(images: &Tensor) -> Result<Tensor> {
    ((images.to_dtype(DType::F32)?.clamp(-1f32, 1f32)? + 1.)? * 127.5)?
        .round()?
        .to_dtype(DType::U8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn empirical_mu_branches() {
        // Above 4300 tokens the shift only depends on the token count.
        assert_eq!(empirical_mu(5000, 4), empirical_mu(5000, 50));
        assert!((empirical_mu(5000, 4) - (0.00016927 * 5000. + 0.45666666)).abs() < 1e-12);
        // 4300 tokens is still on the interpolated side.
        let at_bound = empirical_mu(4300, 10);
        assert!((at_bound - (8.73809524e-05 * 4300. + 1.89833333)).abs() < 1e-9);
        // Below, it interpolates between the fits at 10 and 200 steps.
        let fit_10 = 8.73809524e-05 * 1024. + 1.89833333;
        let fit_200 = 0.00016927 * 1024. + 0.45666666;
        assert!((empirical_mu(1024, 10) - fit_10).abs() < 1e-9);
        assert!((empirical_mu(1024, 200) - fit_200).abs() < 1e-9);
    }

    fn sigmas(cfg: &SchedulerConfig, steps: usize) -> Vec<f64> {
        Scheduler::new(cfg, steps, 1024).unwrap().sigmas
    }

    #[test]
    fn scheduler_variants() {
        let cfg = SchedulerConfig::default();
        let s = sigmas(&cfg, 4);
        assert_eq!(s.len(), 5);
        assert_eq!((s[0], s[4]), (1., 0.));
        assert!(s.windows(2).all(|w| w[0] > w[1]));
        assert_eq!(sigmas(&cfg, 1), [1., 0.]);

        // The linear time shift: mu / (mu + 1/t - 1).
        let linear = SchedulerConfig {
            time_shift_type: "linear".into(),
            ..Default::default()
        };
        let mu = empirical_mu(1024, 2);
        let expected = mu / (mu + (1. / 0.5 - 1.));
        assert!((sigmas(&linear, 2)[1] - expected).abs() < 1e-6);

        // A fixed shift: shift * t / (1 + (shift - 1) t).
        let fixed = SchedulerConfig {
            use_dynamic_shifting: false,
            shift: 3.,
            ..Default::default()
        };
        assert!((sigmas(&fixed, 2)[1] - 3. * 0.5 / (1. + 2. * 0.5)).abs() < 1e-6);
        let identity = SchedulerConfig {
            use_dynamic_shifting: false,
            ..Default::default()
        };
        assert_eq!(sigmas(&identity, 2), [1., 0.5, 0.]);

        // shift_terminal stretches the schedule so that it ends at the terminal value.
        let terminal = SchedulerConfig {
            use_dynamic_shifting: false,
            shift_terminal: Some(0.1),
            ..Default::default()
        };
        let s = sigmas(&terminal, 3);
        assert!((s[2] - 0.1).abs() < 1e-6, "{s:?}");
        assert_eq!(s[0], 1.);
    }

    #[test]
    fn scheduler_rejects_unsupported_configs() {
        let cases = [
            SchedulerConfig {
                invert_sigmas: true,
                ..Default::default()
            },
            SchedulerConfig {
                stochastic_sampling: true,
                ..Default::default()
            },
            SchedulerConfig {
                use_karras_sigmas: true,
                ..Default::default()
            },
            SchedulerConfig {
                use_exponential_sigmas: true,
                ..Default::default()
            },
            SchedulerConfig {
                use_beta_sigmas: true,
                ..Default::default()
            },
            SchedulerConfig {
                time_shift_type: "cosine".into(),
                ..Default::default()
            },
        ];
        for cfg in cases {
            assert!(Scheduler::new(&cfg, 4, 16).is_err(), "{cfg:?}");
        }
        assert!(Scheduler::new(&SchedulerConfig::default(), 0, 16).is_err());
    }

    #[test]
    fn scheduler_config_defaults() {
        let cfg: SchedulerConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.num_train_timesteps, 1000);
        assert_eq!(cfg.shift, 1.);
        assert_eq!(cfg.time_shift_type, "exponential");
        assert!(
            !cfg.use_dynamic_shifting,
            "diffusers defaults to no dynamic shifting"
        );
        assert!(SchedulerConfig::default().use_dynamic_shifting);
    }

    #[test]
    fn euler_step() -> Result<()> {
        let s = Scheduler {
            sigmas: vec![1., 0.75, 0.],
        };
        assert_eq!(s.len(), 2);
        assert!(!s.is_empty());
        let dev = Device::Cpu;
        let x = Tensor::new(&[1f32, 2.], &dev)?;
        let v = Tensor::new(&[4f32, -4.], &dev)?;
        assert_eq!(s.step(&x, &v, 0)?.to_vec1::<f32>()?, [0., 3.]);
        assert_eq!(s.step(&x, &v, 1)?.to_vec1::<f32>()?, [-2., 5.]);
        // The step keeps the latents' dtype.
        let x16 = x.to_dtype(DType::BF16)?;
        assert_eq!(s.step(&x16, &v, 0)?.dtype(), DType::BF16);
        assert!(Scheduler { sigmas: vec![0.] }.is_empty());
        Ok(())
    }

    #[test]
    fn packing_round_trips() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::arange(0f32, 2. * 4. * 4. * 6., &dev)?.reshape((2, 4, 4, 6))?;
        let packed = pack(&x)?;
        assert_eq!(packed.dims(), &[2, 24, 4]);
        // Token (h, w) holds the channels of that position.
        assert_eq!(
            packed.get(0)?.get(7)?.to_vec1::<f32>()?,
            x.get(0)?
                .narrow(1, 1, 1)?
                .narrow(2, 1, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        assert_eq!(
            unpack(&packed, 4, 6)?.flatten_all()?.to_vec1::<f32>()?,
            x.flatten_all()?.to_vec1::<f32>()?
        );
        let p = patchify(&x)?;
        assert_eq!(p.dims(), &[2, 16, 2, 3]);
        // Channel c * 4 + 2 * dy + dx of patch (i, j) is pixel (2i + dy, 2j + dx) of channel c.
        let v = p.to_dtype(DType::F32)?;
        let px = |b: usize, c: usize, h: usize, w: usize| -> Result<f32> {
            x.get(b)?.get(c)?.get(h)?.get(w)?.to_scalar::<f32>()
        };
        let pv = |b: usize, c: usize, h: usize, w: usize| -> Result<f32> {
            v.get(b)?.get(c)?.get(h)?.get(w)?.to_scalar::<f32>()
        };
        assert_eq!(pv(1, 2 * 4 + 3, 1, 2)?, px(1, 2, 3, 5)?);
        assert_eq!(pv(0, 1, 0, 1)?, px(0, 0, 0, 3)?);
        assert_eq!(
            unpatchify(&p)?.flatten_all()?.to_vec1::<f32>()?,
            x.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn cfg_and_image_conversion() -> Result<()> {
        let dev = Device::Cpu;
        let cond = Tensor::new(&[3f32, 1.], &dev)?;
        let uncond = Tensor::new(&[1f32, 1.], &dev)?;
        assert_eq!(apply_cfg(&cond, &uncond, 2.)?.to_vec1::<f32>()?, [5., 1.]);
        let img = Tensor::new(&[-2f32, -1., 0., 1., 3.], &dev)?;
        assert_eq!(to_u8_image(&img)?.to_vec1::<u8>()?, [0, 0, 128, 255, 255]);
        let h = [
            Tensor::new(&[[[1f32, 2.]]], &dev)?,
            Tensor::new(&[[[3f32, 4.]]], &dev)?,
        ];
        assert_eq!(
            stack_text_embeddings(&h)?.to_vec3::<f32>()?,
            [[[1., 2., 3., 4.]]]
        );
        Ok(())
    }

    #[test]
    fn vae_config() {
        let cfg: VaeConfig = serde_json::from_str("{}").unwrap();
        let d = VaeConfig::default();
        assert_eq!(cfg.block_out_channels, [128, 256, 512, 512]);
        assert_eq!(d.block_out_channels, [128, 256, 512, 512]);
        assert_eq!(
            (
                cfg.layers_per_block,
                cfg.latent_channels,
                cfg.norm_num_groups
            ),
            (2, 32, 32)
        );
        assert!(cfg.use_quant_conv && cfg.use_post_quant_conv && cfg.mid_block_add_attention);
        assert_eq!(cfg.batch_norm_eps, d.batch_norm_eps);
        assert!(cfg.decoder_block_out_channels.is_none());

        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let separate = VaeConfig {
            decoder_block_out_channels: Some(vec![1]),
            ..Default::default()
        };
        assert!(Vae::new(&separate, vb.clone()).is_err());
        let no_attn = VaeConfig {
            mid_block_add_attention: false,
            ..Default::default()
        };
        assert!(Vae::new(&no_attn, vb.clone()).is_err());
        // Matching decoder channels are accepted.
        let same = VaeConfig {
            block_out_channels: vec![8, 8],
            decoder_block_out_channels: Some(vec![8, 8]),
            latent_channels: 4,
            norm_num_groups: 4,
            ..Default::default()
        };
        assert!(Vae::new(&same, vb).is_ok());
    }

    #[test]
    fn fixed_shift_is_monotone() {
        for shift in [0.5, 1., 1.15, 3., 10.] {
            let mut prev = 0.;
            for i in 0..=10_000 {
                let v = fixed_shift(i as f64 / 10_000., shift);
                assert!(
                    (prev..=1.).contains(&v),
                    "shift {shift} step {i}: {v} after {prev}"
                );
                prev = v;
            }
            assert_eq!(prev, 1.);
        }
    }
}

/// Bounded proofs of the schedule arithmetic, run with `cargo kani -p candle-transformers`.
#[cfg(kani)]
mod proofs {
    use super::*;

    /// The fixed shift keeps noise levels in `[0, 1]`, rounding included, for the default shift
    /// and the common shift of 3. A symbolic shift, or ordering two symbolic levels, takes CBMC
    /// beyond hours on the bit-precise f64 divisions; `fixed_shift_is_monotone` tests the order.
    #[kani::proof]
    fn fixed_shift_stays_in_unit_interval() {
        let s: f64 = kani::any();
        kani::assume((0. ..=1.).contains(&s));
        for shift in [1., 3.] {
            assert!((0. ..=1.).contains(&fixed_shift(s, shift)));
        }
    }
}
