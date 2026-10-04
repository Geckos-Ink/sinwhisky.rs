# The Angular Fourier Transform for Obtaining a Continuous Spectrum

**Riccardo Cecchini** — riccardo.cecchini594@edu.unito.it

*30 September 2026* - v1.1 04/10/2026

---

## Abstract

A sampled signal can be extended to a continuous geometric model by fitting a circle through local groups of samples and traversing each accepted arc at constant angular speed. Each arc is then the vertical projection of a rotating vector and is therefore a finite sinusoidal segment with its own centre, radius, phase and angular rate. This paper studies the spectrum of that explicitly constructed continuous curve, rather than the spectrum of the source samples alone. We define the **Angular Fourier Transform (AFT)** as the exact Fourier integral of the piecewise-rotational model. The integral has a closed form for every real angular frequency $\omega$, so the query frequency is continuous and is not tied to the number of source samples. We then present the **Fast Angular Fourier Transform (FAFT)**, an evaluation scheme based on a *rotate-and-add* recurrence. Its elementary phase-rotate-and-add operation is analogous to one used inside Cooley–Tukey, but FAFT is not a radix factorization: its rotations come from the fitted geometry and an arbitrary query frequency. It needs no numerical quadrature, no dense resampling and no transcendental function per rotation. We prove that densely sampling the fitted curve ("zoom") only approaches the AFT and differs from it by spectral images and join terms. We also derive analytic frequency derivatives for locating spectral peaks directly on the continuous frequency axis. On 1,024-sample windows of a real recording, one Hann-weighted AFT query costs 2.4–10.6 µs, agrees with independent quadrature to approximately $10^{-14}$ relative error, and yields a converged list of relevant peaks in 1.4–10.9 ms. The method is not claimed to replace the FFT for a complete fixed-bin spectrum, and the angular rates of individual fitted arcs must not be interpreted as physical signal frequencies.

![SinWhisky](https://geckos.ink/stuff/sinwhisky/figures/sinusoids-circle.png)

---

## 1. Introduction

Let $N$ be the number of source samples, $k$ an integer frequency-bin index, $t$ continuous time measured in source-sample intervals, and $i$ the imaginary unit ($i^2=-1$). The query angular frequency $\omega$ is measured in radians per source sample.

The $N$-point discrete Fourier transform (DFT), and therefore the usual FFT implementation of it, returns coefficients on the grid $2\pi k/N$ [1]. Values of the discrete-time spectrum can also be evaluated away from those bins by zero-padding, direct evaluation or related methods [2], but those operations still describe the sampled sequence. They do not define what the signal does between samples.

This paper starts from an explicit continuous model instead. Local source samples are embedded as points in a two-dimensional plane whose horizontal axis is sample time and whose vertical axis is amplitude. Three non-collinear anchor points determine a unique circumcircle. The connecting arc is traversed at constant angular speed, so interpolation is performed uniformly in **angle**, not uniformly in Cartesian $x$. Collinear anchors are represented by the straight-line limit. Repeating this construction over accepted supports gives a piecewise-rotational curve that passes through, or stays within a declared fitting tolerance of, the source samples.

The transform problem is then precise: **what is the Fourier transform of this continuous piecewise-rotational curve, and can it be evaluated quickly and exactly at an arbitrary real frequency?** Three operations must be kept separate:

1. **Geometric fitting.** Determine circles or line limits from source samples.
2. **Dense evaluation (zoom).** Sample the fitted curve at fractional source positions. This produces a denser discrete sequence, not a transform.
3. **Angular Fourier Transform.** Integrate the fitted continuous curve directly against $e^{-i\omega t}$.

Section 2 defines the geometric model without relying on any external implementation. Section 3 defines the AFT and derives its closed form. Section 4 proves the relation between dense evaluation and the AFT. Section 5 presents the FAFT algorithm and Section 6 the search for relevant frequencies. Section 7 reports measurements and Section 8 discusses limitations.

---

## 2. Geometric signal model

### 2.1 Circle determination and aperture

Let $P_j=(x_j,y_j)$, $j=0,1,2$, be three points that are not collinear, with horizontal coordinates $x_j$ and amplitudes $y_j$. Write $C=(c_x,c_y)$ for the circle centre, $r$ for its radius, and $\lVert\cdot\rVert$ for Euclidean distance. The centre of the unique circle through the points solves

<!--
$$ 

\begin{equation}
\begin{pmatrix} 
2(x_1-x_0) & 2(y_1-y_0) \cr 
2(x_2-x_1) & 2(y_2-y_1) 
\end{pmatrix}
\begin{pmatrix} 
c_x \cr 
c_y 
\end{pmatrix}
=
\begin{pmatrix} 
x_1^2-x_0^2+y_1^2-y_0^2 \cr 
x_2^2-x_1^2+y_2^2-y_1^2 
\end{pmatrix},
\qquad r=\|P_1-C\|.
\end{equation}
Thus, we obtain the required center and radius.

$$
-->

<img src="https://geckos.ink/stuff/sinwhisky/figures/equation_1.svg"/>

When the determinant vanishes, the zero-curvature limit is a straight line. We keep that limit explicitly rather than representing it with an infinite radius.

The horizontal coordinate is sample time multiplied by an *aperture* $\alpha$. The aperture controls the relative scale of the horizontal and vertical coordinates and therefore changes the curvature and angular rate of the fitted circle. For the elementary construction from three consecutive samples $y_0,y_1,y_2$, denote their aperture by $\alpha_{\mathrm{triplet}}$. A data-adaptive choice is

$$
\alpha_{\mathrm{triplet}}
=\max\bigl(|y_0-y_1|,\ |y_1-y_2|,\ |y_0-y_2|\bigr).
$$

For a non-flat triplet this gives the points

$$
P_0=(0,y_0),\qquad P_1=(\alpha_{\mathrm{triplet}},y_1),\qquad
P_2=(2\alpha_{\mathrm{triplet}},y_2).
$$

A flat or collinear triplet is handled by the line limit. The AFT derivation below does **not** depend on this particular aperture rule: it only requires the final fitted geometry to be known. For a more general support, let $x[n]$ denote the source amplitude at integer sample index $n$ (distinct from the horizontal point coordinate $x_j$). Let $a$ be the starting sample index, $L>0$ the support length in source-sample intervals, and $m$ the interior anchor offset, with $0<m<L$. With any declared positive aperture $\alpha$, we use

$$
P_0=(0,\,x[a]),\quad P_1=(m\alpha,\,x[a+m]),\quad P_2=(L\alpha,\,x[a+L]).
$$

This separation is useful in practice: the geometric fitting procedure may choose or adapt $\alpha$, while the transform operates only on the resulting arc parameters.

### 2.2 Equal-angle traversal

Let $\theta_j$ be the angle of the radius from $C$ to $P_j$, computed as $\theta_j=\operatorname{atan2}(y_j-c_y,\,x_j-c_x)$ and unwrapped in the direction that visits $P_1$ between $P_0$ and $P_2$. The defining interpolation rule is **equal-angle traversal**. Angle, rather than Cartesian $x$, changes linearly with the source-sample coordinate. Here $u$ is time measured from the start of each side, $y(u)$ is its interpolated amplitude, and $\beta_1$ and $\beta_2$ are the angular rates on the first and second sides. On the first side of the interior anchor, $0\le u<m$, and on the second, $0\le u<L-m$:

$$
y(u)=c_y+r\sin(\theta_0+\beta_1 u),\qquad
\beta_1=\frac{\theta_1-\theta_0}{m};
\qquad
y(u)=c_y+r\sin(\theta_1+\beta_2 u),\qquad
\beta_2=\frac{\theta_2-\theta_1}{L-m}.
$$

Equal increments in $u$ therefore correspond to equal angular increments, and hence to equal arc-length increments on a fixed circle. Except in special geometries, this is not the same curve parameterization obtained by evaluating a semicircle at uniformly spaced Cartesian $x$ coordinates.

Each side is a **rotation**: the vertical projection of a vector of length $r$ turning at $\beta$ radians per source sample around a centre at height $c_y$.

**Definition 1 (rotation).** A rotation is a tuple $(s,L,c,r,\varphi,\beta,d)$ with support $[s,s+L)$, $L>0$. Here $s$ is its starting source-sample position, $L$ its duration, $c$ the vertical centre, $r$ the radius, $\varphi$ the angle at the start, $\beta$ the angular rate, and $d$ a linear amplitude slope per source-sample interval. At position $t$, its amplitude is

$$
y(t)=c+r\sin\bigl(\varphi+\beta\,(t-s)\bigr)+d\,(t-s),
\qquad s\le t<s+L.
$$

Circular arcs have $d=0$; straight-line pieces have $r=0$.

### 2.3 From local fits to a piecewise-rotational curve

A simple dense-resampling construction can fit one circle to every consecutive triplet, evaluate each circle by the equal-angle rule, and blend overlapping predictions with a smooth sine weight. If $z$ new samples are inserted between adjacent source samples and $q=z+1$, a triplet produces $2q+1$ local values. Let $j$ index those local values and $w_j$ be the weight of the $j$-th prediction. One finite-support blending rule is

$$
w_j=\sin\!\left(\frac{\pi(j+1)}{2q+2}\right),
\qquad j=0,\ldots,2q,
$$

with overlapping predictions accumulated as weighted sums and then normalized by the accumulated weight. Original source samples can be restored exactly at their integer positions. This is one way to synthesize a smooth dense sequence, but the transform developed here does not require that particular overlap rule.

For the AFT, the input is the **final continuous curve** after the fitting or selection procedure has been resolved.

**Definition 2 (circle curve).** A circle curve on $[0,N)$ is a finite sequence of rotations whose supports tile $[0,N)$. Rotations with an identical support may be summed ("stacked").

The experiments in Section 7 use hierarchical fitting with variable supports; a triplet-based curve is retained as an independent validation case. A fit is accepted only when the curve reproduces every covered source sample within the declared tolerance. The angular rate $\beta$ is a geometric parameter: it depends on the aperture and on the fitted circle and is not, in general, a physical frequency of the recorded signal (Section 8).

---

## 3. The Angular Fourier Transform

### 3.1 Definition

**Definition 3 (AFT).** Let $y$ be a circle curve, $[p,q)\subseteq[0,N)$ a window with start $p$, end $q$, and duration $T=q-p$, and $w(t)$ a real window weight. Here $q$ denotes the window endpoint, rather than the dense-sampling count used in Section 2.3. The phase factor $e^{-i\omega(t-p)}$ uses $p$ as its time origin. The complex coefficient $\mathcal A[y](\omega)$ of the Angular Fourier Transform at real angular frequency $\omega$ is

$$
\mathcal A[y](\omega)=\int_p^q y(t)\,w(t)\,e^{-i\omega (t-p)}\,dt .
$$

Let $F_s$ be the source sampling rate in samples/s and $f$ the physical frequency in Hz; then $f=\omega F_s/(2\pi)$. We use $w\equiv1$ (rectangular) or the continuous periodic Hann window [3], $w(t)=\tfrac12-\tfrac12\cos\bigl(2\pi(t-p)/T\bigr)$. The normalized coefficient is $\mathcal A/\!\int_p^q w(t)\,dt$. With $A$ denoting a sinusoid's amplitude, its normalized coefficient has magnitude about $A/2$.

Two properties distinguish $\mathcal A$ from the DFT. First, $\omega$ ranges over all of $\mathbb R$, the set of real numbers; there is no grid and no periodicity in $\omega$. Second, the transform is that of the continuous curve, including its behaviour between samples.

### 3.2 Closed form

Let $u$ be local time on a support of duration $L$, $\delta$ a real angular frequency offset, and $k$ a nonnegative integer moment order. Define $E_k(\delta,L)$ as the complex exponential moment of order $k$:

$$
E_k(\delta,L)=\int_0^L u^k e^{i\delta u}\,du,\qquad
E_0(\delta,L)=\frac{e^{i\delta L}-1}{i\delta},\quad
E_k(\delta,L)=\frac{L^k e^{i\delta L}-k\,E_{k-1}(\delta,L)}{i\delta}\ (\delta\ne0),
$$

with $E_k(0,L)=L^{k+1}/(k+1)$; the recurrence applies for $k\ge1$. For a rotation starting at $s$, let $y_0=y(s)$ be its start amplitude and $a=r\sin\varphi$, $b=r\cos\varphi$ its two projected radius components. Here $a$ is an amplitude coefficient, rather than the source index used in Section 2.1. With $u=t-s$, the *anchored* form is

$$
y(s+u)=y_0+a\,(\cos\beta u-1)+b\sin\beta u+d\,u .
$$

**Proposition 1.** For a rectangular window containing the support of one rotation, let $\Phi(\omega)$ be that rotation's Fourier contribution with phase origin at its start $s$. In the expression below, $E_k(\delta)$ abbreviates $E_k(\delta,L)$ for this rotation's duration:

$$
\Phi(\omega)=\int_0^L y(s+u)\,e^{-i\omega u}\,du
= y_0E_0(-\omega)
+a\Bigl[\tfrac12\bigl(E_0(\beta-\omega)+E_0(-\beta-\omega)\bigr)-E_0(-\omega)\Bigr]
+\frac{b}{2i}\bigl[E_0(\beta-\omega)-E_0(-\beta-\omega)\bigr]
+d\,E_1(-\omega).
$$

For a circle curve, let $\Phi_s$ denote each rotation's local contribution (including each stacked rotation), with $s$ its start position. Relative to the window origin $p$, the rectangular transform is

$$
\mathcal A[y](\omega)=\sum_{\text{rotations}} e^{-i\omega (s-p)}\,\Phi_s(\omega).
$$

*Proof.* Write $\cos\beta u=\tfrac12(e^{i\beta u}+e^{-i\beta u})$ and $\sin\beta u=\tfrac1{2i}(e^{i\beta u}-e^{-i\beta u})$. Each resulting term is $\int_0^L u^k e^{i(\pm\beta-\omega)u}\,du=E_k(\pm\beta-\omega)$. Linearity over the tiling and the shift $t=s+u$ give the sum. *Q.E.D.*

The anchored form keeps the start value $y_0$ separate from the deviation. The large and nearly cancelling terms $c$ and $r$ of flat, large-radius arcs never appear in the sum.

**Proposition 2 (Hann weighting).** Let $\Delta=2\pi/T$ be the angular-frequency spacing associated with the window duration, $\mathcal A_{\mathrm{rect}}$ the rectangular transform ($w\equiv1$), and $\mathcal A_{\mathrm{Hann}}$ the Hann-weighted transform of the same curve on the same window. Then

$$
\mathcal A_{\mathrm{Hann}}(\omega)=\tfrac12\mathcal A_{\mathrm{rect}}(\omega)
-\tfrac14\mathcal A_{\mathrm{rect}}(\omega-\Delta)-\tfrac14\mathcal A_{\mathrm{rect}}(\omega+\Delta).
$$

*Proof.* $-\tfrac12\cos\Delta(t-p)=-\tfrac14\bigl(e^{i\Delta(t-p)}+e^{-i\Delta(t-p)}\bigr)$ shifts the frequency by $\mp\Delta$. *Q.E.D.*

The window is therefore applied *after* fitting and exactly. No refit and no sampling occur.

**Proposition 3 (derivatives).** With phase origin at the window centre $t_c=p+T/2$, let $\sigma=s-t_c$ be a rotation's start offset from that centre. For nonnegative integer order $k$, let $\Psi_k$ be its amplitude-weighted Fourier moment, $\Psi_k(\omega)=\int_0^L u^k\,y(s+u)\,e^{-i\omega u}\,du$ (obtained from Proposition 1 with $E_0\to E_k$ and $E_1\to E_{k+1}$). Then

$$
\frac{d}{d\omega}\Bigl[e^{-i\omega\sigma}\Psi_0\Bigr]=-i\,e^{-i\omega\sigma}\bigl(\sigma\Psi_0+\Psi_1\bigr),\qquad
\frac{d^2}{d\omega^2}\Bigl[e^{-i\omega\sigma}\Psi_0\Bigr]=-e^{-i\omega\sigma}\bigl(\sigma^2\Psi_0+2\sigma\Psi_1+\Psi_2\bigr).
$$

Thus $\mathcal A$, $\mathcal A'$ and $\mathcal A''$ are all closed forms of the same rotations; primes denote first and second derivatives with respect to $\omega$.

---

## 4. Dense evaluation is sampling, not the transform

Let *zoom* denote $z$-fold dense evaluation of the fitted continuous curve, i.e. sampling it at $z$ points per source interval. The DFT of those dense samples, divided by $z$, is sometimes used as an approximation to the spectrum of the fitted curve. It is not the AFT.

**Proposition 4.** Let $g(t)=y(t)w(t)$ on $[p,q)$, zero elsewhere, with $w$ the Hann window. Let $y$ have finitely many value jumps at join points $t_j$ on the sampling grid, indexed by $j$, and define $J_j=y(t_j^+)-y(t_j^-)$ as the right-minus-left jump of the unweighted curve. The corresponding jump of $g$ is $J_jw(t_j)$. Let $z$ now denote the positive integer number of samples per source interval (rather than the number of inserted samples in Section 2.3), $n$ an integer sample index, and $m$ an integer spectral-image index. With samples taken from the right, and $\mathbb Z$ denoting all integers, the relation is

$$
\frac1z\sum_{n}g\!\left(p+\tfrac nz\right)e^{-i\omega n/z}
=\sum_{m\in\mathbb Z}\mathcal A_{\mathrm{Hann}}[y]\bigl(\omega+2\pi z m\bigr)
+\frac1z\sum_j \tfrac12 J_j\,w(t_j)\,e^{-i\omega(t_j-p)} .
$$

*Proof.* The Poisson summation formula for a piecewise smooth, compactly supported function applies. Its sampled side uses the mid-value $\tfrac12(g(t^-)+g(t^+))$ at each jump; right-sided sampling adds $\tfrac12J_jw(t_j)$. *Q.E.D.*

The zoomed spectrum is therefore the AFT plus **images** at multiples of $2\pi z$, plus join terms. At $z=1$ the band above the source Nyquist is a mirror of the audio band (Figure 1). As $z\to\infty$ the images vanish like $z^{-2}$ (for a curve that is continuous with kinks) and the join terms like $z^{-1}$. The AFT is the limit, computed directly.

![Figure 1](https://geckos.ink/stuff/sinwhisky/figures/faft-circle-arcs-20260930/faft-zoom-vs-transform.svg)

*Figure 1. One fitted window of a Beethoven recording (60 s), Hann applied after fitting. The exact AFT (blue) against the zoom-1 (orange) and zoom-4 (aqua) sampled transforms of the same curve, up to three times the source Nyquist. Zoom 1 mirrors the band around 44.1 kHz; zoom 4 follows the AFT closely in this range. The AFT has no images.*

---

## 5. The Fast Angular Fourier Transform

### 5.1 The rotate-and-add rule

The Cooley–Tukey FFT repeatedly combines partial transforms by multiplying a partial result by a phase factor and adding it to another [1]. FAFT uses the same elementary **rotate-and-add** operation, but the analogy stops there: FAFT is a linear recurrence over fitted rotations, not a radix decomposition. Its phase factors are determined by the circle geometry and by an arbitrary real query frequency $\omega$, rather than by a fixed set of $N$-th roots of unity.

In this subsection and Algorithm 2, let $q=e^{-i\omega}$ denote the complex phase advance per source interval; it is distinct from the window endpoint. Let $k$ index consecutive supports, with start $s_k$ and duration $L_k$. For consecutive rotations, $e^{-i\omega s_{k+1}}=e^{-i\omega s_k}q^{L_k}$, so support phases are *chained* by multiplication. The terms $e^{i(\pm\beta-\omega)L}$ in $E_0$ factor as follows, where the overline denotes complex conjugation:

$$
e^{i(\beta-\omega)L}=\underbrace{e^{i\beta L}}_{\text{prepared once}}\;q^{L},
\qquad
e^{i(-\beta-\omega)L}=\overline{e^{i\beta L}}\;q^{L}.
$$

The *turn* $e^{i\beta L}$ belongs to the circle and does not depend on $\omega$. One query therefore needs a single $\sin/\cos$ pair (for $q$), one power $q^{L}$ per distinct support length, and per rotation only complex products and two real divisions ($1/(\beta-\omega)$ and $1/(-\beta-\omega)$).

### 5.2 Numerical stability

Let $x=\delta L$ be the dimensionless accumulated frequency offset on a support, and $\varepsilon$ the floating-point machine precision. Here $x$ is a scalar phase offset, distinct from the source sequence $x[n]$ in Section 2.1. When $|x|$ is small (near DC or resonance, $\omega\approx\pm\beta$), the ratio $(e^{ix}-1)/(i\delta)$ cancels. For $|x|$ below a threshold (0.05 for values, 0.5 for derivatives), the implementation evaluates the following series, where $n$ is a nonnegative integer term index and $n!$ is its factorial:

$$
E_k(\delta,L)=L^{k+1}\sum_{n\ge0}\frac{(ix)^n}{n!\,(n+k+1)}
$$

in Horner form in $x^2$ with constant coefficients. Above the threshold, the ratio and recurrence lose at most about $\varepsilon/|x|^{k+1}$. Phases chained by multiplication are re-anchored with an exact $\sin/\cos$ every 64 supports.

### 5.3 Algorithms

In the pseudocode, $o$ is the crop offset from the original rotation start, $s'$ and $L'$ are the cropped start and duration, and `turn` is the prepared complex angular advance over that duration. In Algorithm 2, $Q[L]$ stores the phase advance over duration $L$, $S$ accumulates the transform, `phase` locates the current support relative to the window start, and $e_{\mathrm{dc}},e_+,e_-$ are exponential moments at $-\omega,\beta-\omega,-\beta-\omega$, respectively. `conj` denotes complex conjugation. $R$ is the number of rotations in the prepared window and $K$ the number of queried frequencies.

```text
Algorithm 1  PREPARE(curve, window [p,q), weighting)
  for each rotation (s, L, c, r, φ, β, d) overlapping [p,q):
      crop to [max(s,p), min(s+L,q));   o ← max(s,p) − s
      y0 ← y(s+o);  a ← r·sin(φ+βo);  b ← r·cos(φ+βo)        ▷ anchored form
      turn ← e^{iβL'}                                          ▷ ω-independent
      store (s'−p, L', y0, a, b, β, d, turn)
  lengths ← sorted distinct L'
  return prepared window
```

```text
Algorithm 2  RECT(prepared, ω)                     ▷ Proposition 1, O(R)
  q ← e^{−iω};  Q[L] ← q^L for L in lengths        ▷ one sin/cos, few products
  S ← 0;  phase ← 1
  for each rotation k in time order:
      e_dc ← E0(−ω, L_k; Q[L_k])
      Φ ← y0·e_dc
      if arc:  e₊ ← E0(β−ω; turn·Q[L_k]);   e₋ ← E0(−β−ω; conj(turn)·Q[L_k])
               Φ ← Φ + a·((e₊+e₋)/2 − e_dc) + b·(e₊−e₋)/(2i)
      if d≠0:  Φ ← Φ + d·E1(−ω; Q[L_k], e_dc)
      S ← S + phase·Φ                              ▷ rotate and add
      phase ← phase·Q[L_k]                         ▷ re-anchor every 64 steps
  return S

Algorithm 3  AFT(prepared, ω)
  if rectangular: return RECT(ω)/T
  Δ ← 2π/T
  return (½RECT(ω) − ¼RECT(ω−Δ) − ¼RECT(ω+Δ)) / (T/2)   ▷ Proposition 2
```

**Complexity.** Preparation is $O(R)$ for $R$ rotations in the window. One frequency costs $O(R)$, independent of how many samples the curve could be zoomed to. $K$ arbitrary frequencies cost $O(RK)$. For a dense grid with $K\sim N$ and $R\propto N$, this is quadratic. It is not an alternative to the FFT for computing all DFT bins of the samples, which is a different quantity.

---

## 6. Relevant frequencies on a continuous axis

The AFT is a smooth function of $\omega$. Its relevant frequencies are the local maxima of $|\mathcal A|$ above a floor. They form a variable-length list of real numbers, not $N/2+1$ bins.

The resolution of the transform is set by the window *duration*, $2\pi/T$, not by the number of samples. A zoomed curve has the same $T$ and therefore the same resolution. Seeds are taken on a grid of spacing $2\pi/(T\rho)$ with density $\rho>0$ (default 2), the number of seed spacings per interval of width $2\pi/T$. Each seed is refined by Newton's method on the power $g(\omega)=|\mathcal A(\omega)|^2$. Here $g$ is spectral power, distinct from the time-domain function in Proposition 4; primes denote frequency derivatives, $|\cdot|$ complex magnitude, and $\mathrm{Re}$ the real part. Using Proposition 3:

$$
g'=2\,\mathrm{Re}\bigl(\overline{\mathcal A}\,\mathcal A'\bigr),\qquad
g''=2\bigl(|\mathcal A'|^2+\mathrm{Re}(\overline{\mathcal A}\,\mathcal A'')\bigr),\qquad
\omega\leftarrow\omega-\frac{g'}{g''}\quad(g''<0).
$$

In Algorithm 4, $\omega_{\mathrm{lo}}$ and $\omega_{\mathrm{hi}}$ bound the search band, $j$ indexes scan frequencies, `floor` is the minimum accepted magnitude, and `lo` and `hi` bracket a candidate peak. $A,A',A''$ are the transform and its frequency derivatives, $g_1,g_2$ the first and second power derivatives, and $\omega_{\mathrm{new}}$ the proposed next query frequency.

```text
Algorithm 4  PEAKS(prepared, [ω_lo, ω_hi], ρ, floor)
  scan |AFT| on ω_j = ω_lo + j·2π/(Tρ)          ▷ Hann reuses one RECT grid
  for each local maximum j above floor:
      ω ← parabolic vertex of log|AFT| at j−1, j, j+1
      [lo, hi] ← [ω_{j−1}, ω_{j+1}]
      repeat                                     ▷ safeguarded Newton
          (A, A', A'') ← DERIVATIVES(ω)          ▷ Proposition 3
          g1 ← 2Re(conj(A)A');  g2 ← 2(|A'|² + Re(conj(A)A''))
          if g1 > 0: lo ← ω else hi ← ω
          ω_new ← ω − g1/g2 if g2 < 0 and inside [lo,hi] else (lo+hi)/2
      until |ω_new − ω| < 10⁻¹²
      drop if pinned to a bracket edge (a flank) or not converged
  merge peaks closer than π/(2T), keeping the strongest; apply floors
  return peaks (ω, A(ω))
```

The grouping, merging and exclusion steps are explicit. Nearby candidates are merged, weak ones are excluded by relative and absolute floors, and flanks and unconverged seeds are discarded and counted. The scan is a seed heuristic, not a completeness proof. Finer query precision also does not separate two tones closer than about $2\pi/T$: an arbitrary query position is not added resolution.

---

## 7. Experiments

**Setup.** A Beethoven recording (44.1 kHz) was decoded to PCM16 and the channels averaged to mono. Excerpts of two seconds at 60, 180 and 300 s were used, each contributing 21 evenly spaced windows of $N=1024$ samples. The circles were fitted hierarchically on variable supports of at most 128 samples, with every source sample within 16 PCM16 units. The Hann window was applied after fitting. Timings are warmed medians on an arm64 laptop, Rust 1.98.1.

**Exactness.** The AFT was compared with an independent Gauss–Legendre quadrature of the exported rotations; the largest relative difference was $1.8\cdot10^{-14}$. All 1,219 reported peaks were confirmed as local maxima by the same quadrature. On a triplet-based equal-angle curve constructed directly from Section 2, the FAFT matched an independent numerical-integral baseline to $6.5\cdot10^{-14}$.

**Zoom convergence (Proposition 4).** The median relative $L^2$ difference between the zoom-$z$ sampled transform and the AFT, below Nyquist, is:

| $z$ | 1 | 2 | 4 | 8 | 16 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| 60 s | $2.1\cdot10^{-3}$ | $4.6\cdot10^{-4}$ | $1.2\cdot10^{-4}$ | $3.9\cdot10^{-5}$ | $1.5\cdot10^{-5}$ | $3.3\cdot10^{-6}$ |
| 180 s | $1.3\cdot10^{-3}$ | $2.9\cdot10^{-4}$ | $7.5\cdot10^{-5}$ | $2.1\cdot10^{-5}$ | $6.9\cdot10^{-6}$ | $1.6\cdot10^{-6}$ |
| 300 s | $1.8\cdot10^{-3}$ | $4.2\cdot10^{-4}$ | $1.1\cdot10^{-4}$ | $3.7\cdot10^{-5}$ | $1.4\cdot10^{-5}$ | $3.0\cdot10^{-6}$ |

The decay is close to $z^{-2}$ at low zoom (images) and approaches $z^{-1}$ at high zoom (join terms), as Proposition 4 predicts.

**Cost.** Per-window medians at $N=1024$:

| | 60 s | 180 s | 300 s |
|---|---:|---:|---:|
| Rotations per window | 247 | 141 | 546 |
| FAFT, one Hann frequency | 4.17 µs | 2.44 µs | 10.61 µs |
| Sampled-curve kernel, rotations (zoom 1) | 7.99 µs | 5.57 µs | 14.43 µs |
| Sampled-curve kernel, closed sums (zoom 1) | 54.22 µs | 34.31 µs | 95.01 µs |
| Numerical-integral AFT, triplet curve | 41.00 µs | 41.06 µs | 41.12 µs |
| FAFT, same triplet curve | 12.22 µs | 11.91 µs | 12.42 µs |
| FAFT relevant peaks (scan + Newton) | 3.10 ms | 1.42 ms | 10.93 ms |
| Adaptive sampled-curve peak search | 27.42 ms | 16.91 ms | 49.44 ms |
| FFT of the source samples (plain / 64× padded) | 5.8 / 280 µs | 5.8 / 280 µs | 5.9 / 281 µs |

Query cost grows linearly with the number of rotations; the sizes 256…4096 gave 0.97, 1.69, 3.51, 7.09 and 15.47 µs per frequency (Figure 2).

![Figure 2](https://geckos.ink/stuff/sinwhisky/figures/faft-circle-arcs-20260930/faft-convergence-cost.svg)

*Figure 2. Left: difference between zoom-sampled transforms and the AFT. Middle: cost of one arbitrary frequency against window size. Right: cost of a list of relevant peaks, with the FFT of the samples as a reference for a different target.*

**Continuous peaks.** The FAFT returned on average 19.2, 13.1 and 25.7 peaks per window. Every one was also found by the adaptive sampled-curve search, which found about 5% more. Those extra peaks are weak maxima next to stronger ones and are missed by the density-2 scan; density 4 recovers most of them. Of the peaks of the 64×-padded FFT of the source samples, 89–94% were recovered, with a median distance of 0.18–0.44 Hz. The bin spacing is 43 Hz; the reported AFT peaks are converged to $10^{-12}$ rad/sample on the continuous axis (Figure 3).

![Figure 3](https://geckos.ink/stuff/sinwhisky/figures/faft-circle-arcs-20260930/faft-peaks.svg)

*Figure 3. The AFT of one window (blue) with its Newton-refined peaks, against the 64×-padded FFT of the source samples (orange). The ticks at the bottom are the fixed $N=1024$ DFT bins; the AFT peaks fall between them.*

---

## 8. Discussion and limitations

**Circle rates are not the spectrum.** A natural hope is that the rates $\beta$ of the circles are themselves the frequencies of the sound, so that grouping them would yield the spectrum without any transform. The measurements do not support this. The rates depend on the aperture used to fit the circles. On a synthetic test they clustered well below the true tone. Seeding the peak search from grouped rates recovered only 41–43% of the peaks and was slower than the scan. The continuous frequencies are obtained from the transform of the circles, through the interference of all rotations, not from the rate of any single one.

**Speed.** One exact continuous query is cheaper than the sampled-curve baselines and 3.3–3.5× cheaper than numerical integration of the same curve. A complete, fixed-bin spectrum of the *samples* remains much faster with an FFT. The AFT's advantage is exactness and freedom of the frequency axis, not asymptotic complexity.

**Model.** The AFT is the transform of the fitted curve. The fit is exact within tolerance at the samples only. Content between samples and above the source Nyquist is produced by the circles, not recovered information. Rounded circles leave small value jumps at joins; they cause the $z^{-1}$ tail of Proposition 4.

**Future work.** Future directions include:

- Rotate-and-add in time: sliding windows that update only the rotations entering and leaving.
- Circle fits with continuous joins.
- Certified bounds for peak completeness.
- Circle determinations whose angular rates carry physical frequency.

---

## 9. Conclusion

The geometric construction of Section 2 turns sampled data into a continuous piecewise-rotational curve. Its Fourier transform, the Angular Fourier Transform, has an exact closed form at every real frequency. The Fast Angular Fourier Transform evaluates that form by rotating and adding the contributions of the fitted arcs, with no quadrature, no dense resampling and no transcendental work per rotation. Dense evaluation is therefore not a route to a more exact version of the same spectrum: it produces a sampled approximation with spectral images and join terms that are absent from the direct AFT. Combined with analytic derivatives, the FAFT yields relevant frequencies on a continuous axis and, in the reported experiments, substantially reduces the cost of the continuous-model baselines. It is not claimed to replace the FFT for computing the complete fixed-bin spectrum of the source samples.

---

## References

1. J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series," *Mathematics of Computation* 19 (1965) 297–301.
2. J. O. Smith III, *Mathematics of the Discrete Fourier Transform*, CCRMA, Stanford (online edition).
3. F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform," *Proceedings of the IEEE* 66 (1978) 51–83.

### Reproducibility notes

The mathematical definition of the method is contained in Definitions 1–3, Propositions 1–4 and Algorithms 1–4. Section 5.2 states the numerical-stability thresholds used for near-resonant evaluation, and Section 7 reports the fitting tolerance, support limit, sample format, window size, implementation language/version and benchmark protocol. Validation uses four independent checks: direct Gauss–Legendre quadrature of the fitted rotations, dense-evaluation convergence, closed-form derivative checks, and same-curve comparison on the triplet-based equal-angle construction. The method can therefore be interpreted and implemented from the manuscript alone.
