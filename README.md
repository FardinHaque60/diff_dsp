# diff_dsp
cse 291p final project. inspired by [DDSP](https://arxiv.org/abs/2001.04643)

applying differentiable programming to signal denoising algorithms

## methods
implementing denoising algorithms in the [loma](https://github.com/BachiLi/loma_public) diff programming language

algorithms:
- finite impulse response (FIR)
- infinite impulse response (IIR)
- 1D CNN
- soft thresholding

## experiments
different types of digital signal data, like audio and cellular, will be passed through the algorithms to see how well the learned weights can denoise them