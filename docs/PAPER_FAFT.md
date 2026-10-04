# The Angular Fourier Transform for Obtaining a Continuous Spectrum

**Riccardo Cecchini** — riccardo.cecchini594@edu.unito.it

*30 September 2026*

---

## Abstract

A sampled signal can be extended to a continuous geometric model by fitting a circle through local groups of samples and traversing each accepted arc at constant angular speed. Each arc is then the vertical projection of a rotating radius vector, that is, a finite sinusoidal segment with its own centre, radius, phase and angular rate. This paper studies the spectrum of that explicitly constructed continuous curve, rather than the spectrum of the source samples alone. Starting from the circle through three points, every parameter of a rotation is derived from elementary geometry: the angular sweeps of the two sides of a circle are twice the inscribed angles of the triangle formed by the three points, and a straight line is the limit of infinite radius. We define the **Angular Fourier Transform (AFT)** as the exact Fourier integral of the piecewise-rotational model. Integrating the rotating radius vector gives a closed form for each arc, which depends only on the centre height, the angular rate and the arc's endpoint values and slopes; the time distance between arcs enters only as a phase, by the shift theorem. The transform is therefore available at every real angular frequency $\omega$, and the query frequency is not tied to the number of source samples. We then present the **Fast Angular Fourier Transform (FAFT)**, which evaluates the closed form by chaining these phases with complex multiplications (*rotate and add*). FAFT is a single-frequency recurrence, closer to Goertzel's algorithm than to the Cooley–Tukey FFT: it shares no work between frequencies, and because the fitted curve is a nonlinear function of the samples, no factorization of the discrete Fourier transform applies. It needs no numerical quadrature, no dense resampling and no transcendental function per rotation. We prove that densely sampling the fitted curve ("zoom") only approaches the AFT and differs from it by spectral images and join terms, whose decay follows from the endpoint form. We also derive analytic frequency derivatives for locating spectral peaks directly on the continuous frequency axis. On 1,024-sample windows of a real recording, one Hann-weighted AFT query costs 2.4–10.6 µs, agrees with independent quadrature to approximately $10^{-14}$ relative error, and yields a converged list of relevant peaks in 1.4–10.9 ms. A list of maxima is not yet a list of components: every peak's height is shared with its neighbours through the window kernel, so selecting maxima one by one is a zero-sum choice in which each missed or spurious peak biases the others. We therefore fit all selected tones jointly, with every inner product in closed form from the kernel and the AFT, refine their frequencies together by Gauss–Newton, and obtain each tone's own height, the residual energy, and a bound on the unexplained spectrum at every real frequency. Across consecutive windows the selection is carried forward without rescanning. On a synthetic test with known partials, this raised recall from 89% to 100%, removed spurious peaks, reduced the 95th-percentile height error from 4.7 to 0.6 dB, and, when tracking, cost a median 3.5 ms per window against 6.4 ms for the scan. The method is not claimed to replace the FFT for a complete fixed-bin spectrum. The angular rates of individual arcs are not physical signal frequencies: they depend on the horizontal scale used for fitting, and the frequencies of the signal emerge only from the interference of all arcs.

![SinWhisky](figures/sinusoids-circle.png)

*The SinWhisky construction [1]: the circle through three consecutive samples ($C_j$, at the instants $A_j$) supplies values ($S$) at new positions ($Z$) between the samples.*

---

## 1. Introduction

The $N$-point discrete Fourier transform (DFT), and the fast Fourier transform (FFT) algorithms that compute it, return $N$ coefficients of $N$ samples on the frequency grid $2\pi k/N$, where $k$ is an integer bin index [2, 3]. Values of the discrete-time Fourier transform can also be evaluated away from those bins, by zero-padding, by direct summation or by a single-frequency recurrence such as Goertzel's algorithm [4], but all of these describe the sampled sequence. They do not define what the signal does between samples.

This paper starts from an explicit continuous model instead. Local source samples are embedded as points in a plane whose horizontal axis is scaled sample time and whose vertical axis is amplitude. Three non-collinear points determine a unique circle. The arc between them is traversed at constant angular speed, so interpolation is uniform in **angle**, not in the horizontal coordinate. Collinear points are represented by the straight-line limit. Repeating this construction over accepted supports gives a piecewise-rotational curve that passes through, or stays within a declared tolerance of, the source samples.

The transform problem is then precise: **what is the Fourier transform of this continuous piecewise-rotational curve, and can it be evaluated quickly and exactly at an arbitrary real frequency?** Three operations must be kept separate:

1. **Geometric fitting.** Determine circles or line limits from source samples (Section 2).
2. **Dense evaluation (zoom).** Sample the fitted curve at fractional source positions. This produces a denser discrete sequence, not a transform (Section 4).
3. **Angular Fourier Transform.** Integrate the fitted continuous curve directly against $e^{-i\omega t}$ (Section 3) and evaluate that integral with FAFT (Section 5).

The derivation proceeds from geometry to spectrum. Section 2 derives the circle, its angles and its angular rates from the three anchor points, shows that each half-arc is a rotating radius vector, obtains the line as the limit of infinite radius, and explains why the rates are not frequencies of the signal. Section 3 integrates one rotation in closed form, gives an equivalent form that depends only on the arc's endpoints, and assembles the transform of the whole curve with the shift theorem. Section 4 relates dense evaluation to the AFT. Section 5 presents the FAFT algorithm. Section 6 treats relevant frequencies: the maxima of the transform, the window kernel that shares each peak's level with its neighbours, a joint closed-form tone fit with a certificate of what remains, and the selection of peaks across consecutive windows. Section 7 reports measurements and Section 8 discusses limitations.

### 1.1 What the AFT is not

The AFT and the DFT answer different questions, and FAFT is not a variant of the FFT. The differences are summarized below and justified in Remark 1 and Section 5.1.

| | DFT, computed by an FFT | AFT, evaluated by FAFT |
|---|---|---|
| Input | $N$ samples $x[n]$ | A fitted continuous curve $y(t)$ made of $R$ rotations |
| Transformed object | The sampled sequence | The curve, including its values between samples |
| Frequency axis | $N$ bins $2\pi k/N$; periodic in $\omega$ with period $2\pi$ | Any real $\omega$; not periodic |
| Dependence on the samples | Linear | Nonlinear: the circles are nonlinear functions of the samples |
| Source of speed | A factorization of the DFT matrix shares partial sums among all $N$ bins: $O(N\log N)$ in total | Closed forms and chained phases at one frequency: $O(R)$ per frequency, nothing shared between frequencies |
| Frequency resolution | About $2\pi/N$, equal to the bin spacing | About $2\pi/T$ for a window of duration $T$; finer queries add no resolution |

### 1.2 Notation

| Symbol | Meaning | Unit |
|---|---|---|
| $N$ | Number of source samples | — |
| $x[n]$ | Source sample at integer index $n$ | amplitude |
| $t$ | Continuous time | source-sample intervals ("samples") |
| $i$ | Imaginary unit, $i^2=-1$ | — |
| $\omega$ | Query angular frequency; $\omega F_s/(2\pi)$ hertz at sampling rate $F_s$ | rad/sample |
| $q$ | $e^{-i\omega}$, the phase advance per source interval | — |
| $\alpha$ | Aperture: horizontal plane units per source interval | amplitude/sample |
| $P_j=(X_j,Y_j)$ | Anchor points in the plane, $j=0,1,2$ | amplitude |
| $C=(X_C,Y_C)$, $r$ | Circle centre and radius | amplitude |
| $D$, $\mathcal S$ | Determinant of the centre equations; signed area of the triangle $P_0P_1P_2$ | amplitude² |
| $\theta_j$, $\gamma_j$ | Angle of the radius from $C$ to $P_j$; interior angle of the triangle at $P_j$ | rad |
| $\Theta_1$, $\Theta_2$ | Signed angular sweeps of the two sides of a circle | rad |
| $s$, $L$ | Start and duration of a rotation's support $[s,s+L)$ | samples |
| $c$, $\varphi$, $\beta$, $d$ | Centre height, start angle, angular rate and linear slope of a rotation | amplitude, rad, rad/sample, amplitude/sample |
| $\zeta(u)$ | Complex radius vector at local time $u$ | amplitude |
| $a$, $b$ | Vertical and horizontal components of the start radius vector, $\zeta(0)=b+ia$ | amplitude |
| $[t_0,t_0+T)$ | Transform window with start $t_0$ and duration $T$ | samples |
| $w(t)$, $W$ | Window weight and its integral | —, samples |
| $E_\ell(\delta,L)$ | Exponential moment of order $\ell$ at frequency offset $\delta$ | samples$^{\ell+1}$ |
| $\Phi(\omega)$ | Transform of one rotation, with phase origin at its start | amplitude·samples |
| $\mathcal A[y] (\omega)$ | AFT of the curve $y$ | amplitude·samples |
| $R$, $K$ | Number of rotations in the window; number of queried frequencies | — |

Both plane coordinates are in amplitude units, because the horizontal coordinate is a sample offset multiplied by $\alpha$. Overlines denote complex conjugation, primes on $y$ denote derivatives with respect to time, and primes on $\mathcal A$ derivatives with respect to $\omega$.

---

## 2. Geometric signal model

### 2.1 Plane embedding and aperture

A support is a range of source samples from $n_0$ to $n_0+L$, with integer start $n_0$, integer length $L\ge2$, and an interior anchor offset $m$ with $0 < m < L$. Its three anchor points are placed in a plane whose horizontal coordinate is the sample offset multiplied by a positive *aperture* $\alpha$:

$$
P_0=\bigl(0,\ x[n_0]\bigr),\qquad P_1=\bigl(m\alpha,\ x[n_0+m]\bigr),\qquad P_2=\bigl(L\alpha,\ x[n_0+L]\bigr). \tag{1}
$$

The aperture sets the relative scale of the two axes. A circle is not preserved by unequal scaling of the axes, so $\alpha$ changes the fitted circle, its curvature and its angular rates (Section 2.6). For the elementary triplet construction ($L=2$, $m=1$), a data-adaptive choice is the largest amplitude difference of the three anchors,

$$
\alpha_{\mathrm{triplet}}=\max\bigl(\lvert Y_0-Y_1\rvert,\ \lvert Y_1-Y_2\rvert,\ \lvert Y_0-Y_2\rvert\bigr), \tag{2}
$$

replaced by $1$ for a constant triplet. We call (2) the *range aperture*. The hierarchical fitter of Section 7 uses $\alpha=2g\alpha_{\mathrm{range}}/L$, where $\alpha_{\mathrm{range}}$ is the same largest difference of the three anchors and $g\in\lbrace 1/2,1,2\rbrace$ is a searched gain. The transform derived in Section 3 does not depend on how $\alpha$ was chosen: it uses only the resulting arc parameters.

### 2.2 The circle through three points

A point $C$ is equidistant from $P_0$ and $P_1$ if and only if $\lVert P_1-C\rVert^2=\lVert P_0-C\rVert^2$. Expanding both squares, the quadratic terms in $C$ cancel and leave the linear equation $2(P_1-P_0)\cdot C=\lVert P_1\rVert^2-\lVert P_0\rVert^2$, the perpendicular bisector of the chord $P_0P_1$. The centre $C=(X_C,Y_C)$ is the intersection of this bisector with that of the chord $P_1P_2$:

$$
2(X_1-X_0)X_C+2(Y_1-Y_0)Y_C=X_1^2-X_0^2+Y_1^2-Y_0^2, \tag{3a}
$$

$$
2(X_2-X_1)X_C+2(Y_2-Y_1)Y_C=X_2^2-X_1^2+Y_2^2-Y_1^2. \tag{3b}
$$

Write $B_1$ and $B_2$ for the right-hand sides of (3a) and (3b). The determinant of the system is

$$
D=4\bigl[(X_1-X_0)(Y_2-Y_1)-(Y_1-Y_0)(X_2-X_1)\bigr]=8\mathcal S, \tag{4}
$$

where $\mathcal S$ is the signed area of the triangle $P_0P_1P_2$, positive when its vertices are in counter-clockwise order. By Cramer's rule,

$$
X_C=\frac{2(Y_2-Y_1)B_1-2(Y_1-Y_0)B_2}{D},\qquad Y_C=\frac{2(X_1-X_0)B_2-2(X_2-X_1)B_1}{D},\qquad r=\lVert P_1-C\rVert. \tag{5}
$$

$D=0$ exactly when the three points are collinear. The circle then degenerates into its zero-curvature limit, a straight line, which is kept explicitly (Section 2.5) and never stored as an infinite radius. In floating point the line limit is used when $\lvert D\rvert$ falls below a declared tolerance.

The radius also follows from the triangle alone. With $\gamma_j$ the interior angle at $P_j$, the law of sines gives [5]

$$
r=\frac{\lVert P_2-P_0\rVert}{2\sin\gamma_1}=\frac{\lVert P_1-P_0\rVert}{2\sin\gamma_2}=\frac{\lVert P_2-P_1\rVert}{2\sin\gamma_0}. \tag{6}
$$

A nearly flat triangle, with $\gamma_1$ close to $\pi$, therefore has a large radius.

### 2.3 Angles, direction and angular sweeps

Let $\theta_j=\operatorname{atan2}(Y_j-Y_C,\ X_j-X_C)$ be the angle of the radius from $C$ to $P_j$, so that

$$
P_j=C+r\bigl(\cos\theta_j,\ \sin\theta_j\bigr). \tag{7}
$$

The angles are unwrapped in the direction that visits $P_1$ between $P_0$ and $P_2$. This direction is counter-clockwise when $D>0$ and clockwise when $D < 0$. With time increasing to the right, $D < 0$ means that $P_1$ lies above the chord $P_0P_2$ (locally concave samples), and $D>0$ that it lies below. The arc traversed from $P_0$ to $P_1$ is the one that does not contain $P_2$. By the inscribed angle theorem [5], its central angle is twice the angle $\gamma_2$ that the chord $P_0P_1$ subtends at $P_2$. The same argument applies to the second arc and $\gamma_0$. The signed angular sweeps of the two sides are therefore

$$
\Theta_1=\theta_1-\theta_0=2\gamma_2\operatorname{sgn}D,\qquad \Theta_2=\theta_2-\theta_1=2\gamma_0\operatorname{sgn}D,\qquad \Theta_1+\Theta_2=2(\pi-\gamma_1)\operatorname{sgn}D. \tag{8}
$$

Each sweep has magnitude between $0$ and $2\pi$, and so does their sum: the curve never makes a full turn. As the triangle flattens, $\gamma_1\to\pi$ and both sweeps tend to zero.

### 2.4 Equal-angle traversal: each half-arc is a rotation

The defining interpolation rule is **equal-angle traversal**: on each side of the interior anchor, the angle changes linearly with time. Let $u$ be time measured from the start of a side, with $0\le u < m$ on the first side and $0\le u < L-m$ on the second. The angle is

$$
\theta(u)=\theta_0+\beta_1u,\quad \beta_1=\frac{\Theta_1}{m}=\frac{2\gamma_2\operatorname{sgn}D}{m};\qquad\qquad \theta(u)=\theta_1+\beta_2u,\quad \beta_2=\frac{\Theta_2}{L-m}=\frac{2\gamma_0\operatorname{sgn}D}{L-m}. \tag{9}
$$

The angular rates $\beta_1$ and $\beta_2$, in radians per source sample, are therefore the inscribed angles of the anchor triangle, doubled and divided by the durations of the sides. The moving point visits $P_0$, $P_1$ and $P_2$ at times $0$, $m$ and $L$ of the support (Figure 1).

Identify the plane with the complex plane, the point $(X,Y)$ with the number $X+iY$. The moving point is the centre plus a rotating radius vector $\zeta$:

$$
X(u)+iY(u)=X_C+iY_C+\zeta(u),\qquad \zeta(u)=re^{i\theta(u)}=\zeta(0)e^{i\beta u}. \tag{10}
$$

The interpolated amplitude is its imaginary part, the vertical projection of the rotating radius vector:

$$
y(u)=Y_C+r\sin\theta(u)=Y_C+\frac{\zeta(0)e^{i\beta u}-\overline{\zeta(0)}e^{-i\beta u}}{2i}. \tag{11}
$$

Equal increments of $u$ correspond to equal angular increments, hence to equal arc lengths $r\lvert\beta\rvert$ per sample. This is not the curve obtained by evaluating the circle at uniformly spaced horizontal coordinates: the horizontal coordinate of the moving point, $X(u)=X_C+r\cos\theta(u)$, is not proportional to time. The horizontal centre $X_C$ and the aperture $\alpha$ influence the curve only through the angles $\theta_j$. Once the angles are known, $y$ depends on $Y_C$, $r$, $\theta_0$ and $\beta$ alone.

The time derivative is the rate times the horizontal component of the radius vector:

$$
y'(u)=r\beta\cos\theta(u)=\beta\bigl(X(u)-X_C\bigr). \tag{12}
$$

At the interior anchor the angle is continuous but the rate changes from $\beta_1$ to $\beta_2$. The curve is therefore continuous at $P_1$, and its slope jumps by $r\cos\theta_1(\beta_2-\beta_1)$ unless $\beta_1=\beta_2$ or $P_1$ is the highest or lowest point of the circle ($\cos\theta_1=0$).

![Figure 1](figures/faft-three-points.svg)

*Figure 1. From three anchor points to two rotations. Left: in the plane, the circle through $P_0$, $P_1$, $P_2$; the central angles $\Theta_1$ and $\Theta_2$ are twice the inscribed angles $\gamma_2$ and $\gamma_0$, as in (8). Right: traversing the same arc at constant angular speed on each side gives two sinusoidal pieces around the centre height $c=Y_C$, with rates $\beta_1=\Theta_1/m$ and $\beta_2=\Theta_2/(L-m)$; the slope changes at $P_1$. Aqua dots mark half-sample time steps, which are equal angle steps on each side. Example: $L=5$, $m=2$, $\alpha=0.32$, anchor amplitudes $0.05$, $1.0$, $0.45$; the triangle is clockwise ($D < 0$), so both rates are negative ($-0.765$ and $-0.489$ rad/sample).*

**Definition 1 (rotation).** A rotation is a tuple $(s,L,c,r,\varphi,\beta,d)$: support $[s,s+L)$ with start $s$ and duration $L>0$ in samples, centre height $c$, radius $r\ge0$, start angle $\varphi$, angular rate $\beta$ and linear slope $d$. At position $t$ its amplitude is

$$
y(t)=c+r\sin\bigl(\varphi+\beta(t-s)\bigr)+d(t-s),\qquad s\le t < s+L. \tag{13}
$$

A half-arc has $c=Y_C$, $\varphi=\theta_0$ or $\theta_1$, $\beta=\beta_1$ or $\beta_2$, and $d=0$; a straight piece has $r=0$. Each circle therefore yields two rotations that share $c$ and $r$ and meet at $P_1$.

**Anchored form.** Expanding $\sin(\varphi+\beta u)$ and writing $y_0=y(s)=c+a$ for the start value gives

$$
y(s+u)=y_0+a(\cos\beta u-1)+b\sin\beta u+du,\qquad a=r\sin\varphi,\quad b=r\cos\varphi. \tag{14}
$$

Geometrically, $b+ia=\zeta(0)$: $a$ is the height of the start point above the centre and $b$ its horizontal offset from the centre. The anchored form starts from the actual start value and adds deviations that vanish at $u=0$, so a flat arc, whose centre lies far away, is evaluated without forming its centre height $c$ (Section 5.3).

### 2.5 The straight line as the limit of infinite radius

Let a rotation's radius grow while its start value $y_0$, its start slope $y'(0)=d$ and its start angle $\varphi$, with $\cos\varphi\ne0$, stay fixed. By (12), $d=\beta b=\beta r\cos\varphi$, so the rate $\beta=d/(r\cos\varphi)$ tends to zero like $1/r$. In (14), $a\beta^2=d^2\sin\varphi/(r\cos^2\varphi)$ and $b\beta=d$, so

$$
a(\cos\beta u-1)=-\tfrac12a\beta^2u^2+O\bigl(a\beta^4u^4\bigr)\to0,\qquad b\sin\beta u=b\beta u+O\bigl(b\beta^3u^3\bigr)\to du, \tag{15}
$$

and the rotation tends to the line $y_0+du$, with an error of order $1/r$. A straight piece is therefore exactly the infinite-radius limit of a rotation, and each formula for arcs in Section 3 tends to the corresponding formula for lines. The implementation stores this limit as $r=0$ with a finite slope $d$, which keeps the line exact and avoids infinite or nearly cancelling parameters.

### 2.6 Why angular rates are not signal frequencies

By (9) the rates are geometric: they are fixed by the shape of the anchor triangle in the plane, and that shape depends on the aperture. A sinusoid makes this exact. Let three consecutive samples be centred on a crest of $A\cos(\omega_0t)$, so that $Y_0=Y_2=A\cos\omega_0$ and $Y_1=A$, placed at $X=0$, $\alpha$ and $2\alpha$. With the drop $h=A(1-\cos\omega_0)$, the triangle is isosceles with $\tan\gamma_0=\tan\gamma_2=h/\alpha$, it is clockwise, and (9) gives

$$
\beta_1=\beta_2=-2\arctan\frac{h}{\alpha}. \tag{16}
$$

The rate matches the frequency of the sinusoid, $\lvert\beta\rvert=\omega_0$, only for the aperture $\alpha=h\cot(\omega_0/2)=A\sin\omega_0$, which is the largest magnitude of the central difference $(x[n+1]-x[n-1])/2$ of the sampled sinusoid. With the range aperture (2), $\alpha=h$ and every symmetric crest is traversed at exactly $-\pi/2$ rad/sample, whatever $\omega_0$. With $\alpha=1$, $A=1000$ and $\omega_0=0.05$, the rate is $-1.79$ rad/sample, about 36 times the frequency of the tone. Angular rates describe local shape in the fitting plane. The frequencies of the signal emerge from the transform of the whole curve, through the interference of all rotations (Remark 2).

### 2.7 From local fits to a circle curve

**Definition 2 (circle curve).** A circle curve on $[0,N)$ is a finite sequence of rotations, in time order, whose supports tile $[0,N)$. Rotations with an identical support may be summed ("stacked").

Two constructions are used in this paper.

*Hierarchical curve.* The fitter of Section 7 chooses variable integer supports, interior anchors and apertures. It keeps a circle only if the two rotations it yields reproduce every covered source sample within a declared tolerance, and otherwise splits the support or uses a line. Consecutive supports share their boundary sample, so the curve is continuous up to the rounding of the stored circle parameters, which leaves small, tolerance-bounded value jumps at joins.

*Triplet curve.* For validation against the original construction, each unit interval $[n,n+1)$ uses the two circles that contain it: the circle through samples $n-1$, $n$, $n+1$ (its second side) and the circle through $n$, $n+1$, $n+2$ (its first side), each fitted with the range aperture (2). The curve on the interval is the equal-weight average of the two sides, that is, two stacked rotations with halved centre heights and radii. A side whose endpoints miss the samples by more than a small relative tolerance is replaced by the line through the two samples, and the frame is closed periodically, from the last sample back to the first.

The triplet curve differs from the dense resampler of the original SinWhisky [1], which blends overlapping equal-angle predictions with a sine weight: with $z$ output positions per source interval, a triplet yields $2z+1$ local values $j=0,\ldots,2z$, weighted by $w_j=\sin\bigl(\pi(j+1)/(2z+2)\bigr)$ and normalized by the accumulated weight. The transform below takes any circle curve as input and depends on neither blending rule.

---

## 3. The Angular Fourier Transform

### 3.1 Definition

**Definition 3 (AFT).** Let $y$ be a circle curve, $[t_0,t_0+T)\subseteq[0,N)$ a window with start $t_0$ and duration $T>0$, and $w$ a real weight on the window. The Angular Fourier Transform of $y$ at a real angular frequency $\omega$ is the complex number

$$
\mathcal A[y] (\omega)=\int_{t_0}^{t_0+T}y(t)w(t)e^{-i\omega(t-t_0)}dt. \tag{17}
$$

The phase origin is the window start $t_0$. We use the rectangular weight $w\equiv1$ or the continuous periodic Hann weight $w(t)=\tfrac12-\tfrac12\cos\bigl(2\pi(t-t_0)/T\bigr)$ [6]. The normalized coefficient is $\mathcal A[y] (\omega)/W$, with $W=\int_{t_0}^{t_0+T}w(t)dt$ equal to $T$ for the rectangular weight and $T/2$ for Hann; a sinusoid of amplitude $A$ then has a normalized magnitude close to $A/2$ at its frequency.

Two properties distinguish $\mathcal A$ from the DFT. First, $\omega$ ranges over all real numbers: there is no frequency grid and no periodicity in $\omega$. Second, the transform is that of the continuous curve, including its behaviour between samples and its content above the source Nyquist frequency $\omega=\pi$, which is produced by the model rather than recorded.

**Remark 1 (linearity).** $\mathcal A$ is linear in the curve $y$, but the curve is a nonlinear function of the samples: the centre (5) is a rational function of the anchor coordinates, and the range aperture (2) itself depends on the samples. The map from samples to $\mathcal A$ is therefore not linear. It has no convolution theorem and cannot be computed as a DFT of the samples followed by a fixed spectral filter. The all-line model is the exception that shows what circles change: if every piece is the line through two consecutive samples of a periodic frame, the curve is the convolution of the samples with a triangular kernel and, at the DFT frequencies, $\mathcal A(2\pi k/N)=\operatorname{sinc}^2(\pi k/N)X[k]$, with $X[k]$ the DFT of the samples and $\operatorname{sinc}v=\sin v/v$ [7]. Circles replace that fixed kernel with shapes that depend on the data.

### 3.2 Exponential moments

For a support of duration $L$, a real frequency offset $\delta$ and an integer order $\ell\ge0$, define the exponential moment

$$
E_\ell(\delta,L)=\int_0^Lu^\ell e^{i\delta u}du. \tag{18}
$$

For $\delta\ne0$, direct integration and integration by parts give

$$
E_0(\delta,L)=\frac{e^{i\delta L}-1}{i\delta},\qquad E_\ell(\delta,L)=\frac{L^\ell e^{i\delta L}-\ell E_{\ell-1}(\delta,L)}{i\delta}\quad(\ell\ge1), \tag{19}
$$

and $E_\ell(0,L)=L^{\ell+1}/(\ell+1)$. Each $E_\ell(\cdot,L)$ is an entire function of $\delta$; the singularity of (19) at $\delta=0$ is removable, and Section 5.3 evaluates the moments by a power series near it. Every closed form below combines these moments at the offsets $-\omega$ and $\pm\beta-\omega$. Where the duration is clear, $E_\ell(\delta)$ abbreviates $E_\ell(\delta,L)$.

### 3.3 Transform of one rotation

Fix one rotation and measure time from its start, $t=s+u$. Its local transform, with phase origin at $s$, is

$$
\Phi(\omega)=\int_0^Ly(s+u)e^{-i\omega u}du. \tag{20}
$$

**Proposition 1 (phasor form).** For a rotation $(s,L,c,r,\varphi,\beta,d)$ with start radius vector $\zeta_0=re^{i\varphi}$,

$$
\Phi(\omega)=cE_0(-\omega)+\frac{\zeta_0E_0(\beta-\omega)-\overline{\zeta_0}E_0(-\beta-\omega)}{2i}+dE_1(-\omega). \tag{21}
$$

*Proof.* Insert (11) and the slope term of (13) into (20). The centre contributes $c\int_0^Le^{-i\omega u}du=cE_0(-\omega)$. The radius vector rotating at $+\beta$ contributes $\zeta_0\int_0^Le^{i(\beta-\omega)u}du=\zeta_0E_0(\beta-\omega)$, and its conjugate, rotating at $-\beta$, contributes $\overline{\zeta_0}E_0(-\beta-\omega)$. The slope contributes $d\int_0^Lue^{-i\omega u}du=dE_1(-\omega)$. $\square$

This decomposition is the geometric content of the transform. A rotation is a constant, its centre height, plus two counter-rotating phasors of length $r/2$, and each phasor resonates where the query frequency matches its own rate, at $\omega=\beta$ and at $\omega=-\beta$. On a finite support a resonance has the finite height $rL/2$ and a main lobe of half-width $2\pi/L$.

**Corollary 1 (anchored form).** With $y_0=c+a$, $a=r\sin\varphi$ and $b=r\cos\varphi$ as in (14), (21) equals

$$
\Phi(\omega)=y_0E_0(-\omega)+a\Bigl[\tfrac12\bigl(E_0(\beta-\omega)+E_0(-\beta-\omega)\bigr)-E_0(-\omega)\Bigr]+\frac{b}{2i}\bigl[E_0(\beta-\omega)-E_0(-\beta-\omega)\bigr]+dE_1(-\omega). \tag{22}
$$

*Proof.* Substitute $c=y_0-a$ and $\zeta_0=b+ia$ in (21) and collect the terms in $a$ and $b$. $\square$

The two bracketed combinations in (22) are the transforms of $\cos\beta u-1$ and $\sin\beta u$. Forms (21) and (22) are equal in exact arithmetic; FAFT evaluates (22), which matches the anchored evaluation (14) of the fitted curve (Section 5.3).

**Corollary 2 (line).** For a straight piece ($r=0$), $\Phi(\omega)=y_0E_0(-\omega)+dE_1(-\omega)$. This is also the limit of (22) along the infinite-radius limit of Section 2.5. Since $\partial E_0(\delta)/\partial\delta=iE_1(\delta)$, the $b$ term tends to $b\beta E_1(-\omega)\to dE_1(-\omega)$, and the $a$ term is $O(a\beta^2)\to0$. $\square$

### 3.4 The endpoint form

A rotation with $d=0$ satisfies, by (13),

$$
y''(u)=-\beta^2\bigl(y(u)-c\bigr), \tag{23}
$$

the equation of a harmonic oscillator of angular frequency $\beta$ centred at height $c$. Integrating (20) by parts twice turns this equation into a statement about the endpoints of the support only.

**Proposition 2 (endpoint form).** For an arc ($d=0$) and $\omega^2\ne\beta^2$, let $y_0$, $y'_0$ be the value and slope at the start of the support and $y_L$, $y'_L$ their left limits at its end. Then

$$
\Phi(\omega)=\frac{\bigl(y'_Le^{-i\omega L}-y'_0\bigr)+i\omega\bigl(y_Le^{-i\omega L}-y_0\bigr)-\beta^2cE_0(-\omega)}{\omega^2-\beta^2}. \tag{24}
$$

*Proof.* Two integrations by parts give $\int_0^Ly''(u)e^{-i\omega u}du=\bigl[y'(u)e^{-i\omega u}\bigr]_0^L+i\omega\bigl[y(u)e^{-i\omega u}\bigr]_0^L-\omega^2\Phi(\omega)$. By (23) the left side equals $-\beta^2\Phi(\omega)+\beta^2cE_0(-\omega)$. Solve for $\Phi(\omega)$. $\square$

By (12) the slopes are $y'=\beta(X-X_C)$, so (24) depends on the circle only through the centre height $c$, the rate $\beta$, and the heights and horizontal offsets of the two endpoints, that is, through the radius vectors $\zeta_0$ and $\zeta_L=\zeta_0e^{i\beta L}$ at the ends of the arc. In terms of these vectors, (21) reads

$$
\Phi(\omega)=cE_0(-\omega)-\frac12\Bigl[\frac{\zeta_Le^{-i\omega L}-\zeta_0}{\beta-\omega}+\frac{\overline{\zeta_L}e^{-i\omega L}-\overline{\zeta_0}}{\beta+\omega}\Bigr], \tag{25}
$$

because $\zeta_0e^{i(\beta-\omega)L}=\zeta_Le^{-i\omega L}$. In words: the spectrum of an arc is generated at its two ends, and everything between them is summarized by the centre height and the angular rate. The resonances at $\omega=\pm\beta$ are removable singularities of (24) and (25), where numerator and denominator vanish together; their values are the finite limits given by (21). A line satisfies (23) with $\beta=0$, and (24) with $\beta=0$ gives its transform for $\omega\ne0$, so the endpoint form covers every piece of a circle curve.

**Corollary 3 (decay at high frequency).** As $\lvert\omega\rvert$ grows beyond all rates, the rectangular AFT of a circle curve satisfies

$$
\mathcal A_{\mathrm{rect}}[y] (\omega)=\sum_{\tau}e^{-i\omega(\tau-t_0)}\Bigl[\frac{J_\tau}{i\omega}+\frac{J'_\tau}{(i\omega)^2}\Bigr]+O\bigl(\lvert\omega\rvert^{-3}\bigr), \tag{26}
$$

where the sum runs over the joins $\tau$ between pieces, and $J_\tau$ and $J'_\tau$ are the jumps (right limit minus left limit) of the value and of the slope at $\tau$. The sum includes the two window ends, where the curve jumps from zero and back to zero.

*Proof.* Multiply (24) for each piece by its phase $e^{-i\omega(s-t_0)}$ from Section 3.5 and sum over the pieces. Since $1/(\omega^2-\beta^2)=\omega^{-2}+O(\omega^{-4})$ and $E_0(-\omega)=O(\omega^{-1})$, the leading terms are the endpoint terms divided by $\omega^2$. At a join $\tau$, the end terms of one piece and the start terms of the next combine into $-e^{-i\omega(\tau-t_0)}\bigl(J'_\tau+i\omega J_\tau\bigr)/\omega^2$, which is the bracket of (26). This is the classical correspondence between the singularities of a function and the asymptotic behaviour of its Fourier transform [8]. $\square$

A continuous circle curve has $J_\tau=0$ at every interior join, so apart from the window ends its spectrum decays like $\omega^{-2}$, with a strength set by the slope jumps (kinks) where adjacent rotations disagree; this includes the interior anchor of every circle (Section 2.4). The tolerance-bounded value jumps of rounded circles add an $\omega^{-1}$ tail. For the Hann weight, (26) applies to $yw$; since $w$ and $w'$ vanish at the window ends, the ends then contribute neither an $\omega^{-1}$ nor an $\omega^{-2}$ term.

### 3.5 Assembling the curve: time distance becomes phase

A rotation that starts at $s$ inside the window is the local function of Section 3.3 delayed by $s-t_0$. By the shift theorem [9], the delay multiplies its transform by a unit phasor:

$$
\int_s^{s+L}y(t)e^{-i\omega(t-t_0)}dt=e^{-i\omega(s-t_0)}\Phi(\omega). \tag{27}
$$

Rotations that overlap a window boundary are first cropped to the window. Cropping a rotation at offset $o$ from its start leaves a rotation with the same centre height, radius and rate, start angle $\varphi+\beta o$, start value $y(s+o)$ and a shorter duration; nothing is refitted.

**Proposition 3 (rectangular AFT of a circle curve).** Let the rotations in the window, after cropping, have starts $s_k$, durations $L_k$ and local transforms $\Phi_k$, for $k=1,\ldots,R$, with stacked rotations listed separately. Then

$$
\mathcal A_{\mathrm{rect}}[y] (\omega)=\sum_{k=1}^Re^{-i\omega(s_k-t_0)}\Phi_k(\omega). \tag{28}
$$

*Proof.* The supports tile the window, so (17) with $w\equiv1$ splits into one integral per support, and stacked rotations add by linearity. Apply (27) to each integral. $\square$

The time distance between two rotations therefore enters only as the relative phase $e^{-i\omega(s_k-s_j)}$. Consecutive supports satisfy $s_{k+1}=s_k+L_k$, so the phases form a chain,

$$
e^{-i\omega(s_{k+1}-t_0)}=e^{-i\omega(s_k-t_0)}q^{L_k},\qquad q=e^{-i\omega}, \tag{29}
$$

which FAFT evaluates by multiplication (Section 5).

**Remark 2 (where spectral peaks come from).** Each local transform $\Phi_k$ is broad: its resonances at $\pm\beta_k$ have a width of order $2\pi/L_k$, and supports are short. The narrow peaks of $\mathcal A$, with a width of order $2\pi/T$, come from the phases (29): at a frequency where the delayed contributions $e^{-i\omega(s_k-t_0)}\Phi_k(\omega)$ align, they add coherently. The case of $M$ identical rotations of duration $L$, placed back to back so that $T=ML$, makes this exact. A geometric series gives

$$
\mathcal A_{\mathrm{rect}}(\omega)=\Phi(\omega)\sum_{k=0}^{M-1}q^{kL}=\Phi(\omega)\frac{1-e^{-i\omega T}}{1-e^{-i\omega L}}\qquad\bigl(e^{-i\omega L}\ne1\bigr), \tag{30}
$$

and $\mathcal A_{\mathrm{rect}}(\omega)=M\Phi(\omega)$ where $e^{-i\omega L}=1$. The transform is the product of a broad *element factor* $\Phi$, set by the shape of one arc, and a narrow *array factor*, a Dirichlet kernel set by the time distances alone. Its narrow peaks sit at the harmonics $\omega=2\pi j/L$ of the spacing, for integer $j$, whatever the rate $\beta$; the rate only weights them through $\Phi$. This is why the rates are not the spectrum (Section 2.6).

### 3.6 Hann weighting

**Proposition 4 (Hann weighting).** Let $\Delta=2\pi/T$. Then

$$
\mathcal A_{\mathrm{Hann}}(\omega)=\tfrac12\mathcal A_{\mathrm{rect}}(\omega)-\tfrac14\mathcal A_{\mathrm{rect}}(\omega-\Delta)-\tfrac14\mathcal A_{\mathrm{rect}}(\omega+\Delta). \tag{31}
$$

*Proof.* $w(t)=\tfrac12-\tfrac14e^{i\Delta(t-t_0)}-\tfrac14e^{-i\Delta(t-t_0)}$, and $e^{\pm i\Delta(t-t_0)}e^{-i\omega(t-t_0)}=e^{-i(\omega\mp\Delta)(t-t_0)}$. $\square$

The window is therefore applied exactly and after fitting: no refit and no sampling occur. Moving the phase origin to the window centre $t_c=t_0+T/2$ defines $\mathcal A^{c}(\omega)=e^{i\omega T/2}\mathcal A(\omega)$, with the same magnitude, for either weight. Because $\Delta T/2=\pi$, the shifted terms change sign:

$$
\mathcal A^{c}_{\mathrm{Hann}}(\omega)=\tfrac12\mathcal A^{c}_{\mathrm{rect}}(\omega)+\tfrac14\mathcal A^{c}_{\mathrm{rect}}(\omega-\Delta)+\tfrac14\mathcal A^{c}_{\mathrm{rect}}(\omega+\Delta). \tag{32}
$$

### 3.7 Frequency derivatives

**Proposition 5 (derivatives).** With phase origin at the window centre, let $\sigma_k=s_k-t_c$ be the offset of rotation $k$ from the centre, and define its amplitude-weighted Fourier moments

$$
\Psi_{k,\ell}(\omega)=\int_0^{L_k}u^\ell y(s_k+u)e^{-i\omega u}du,\qquad \ell=0,1,2, \tag{33}
$$

which follow from (22) by replacing each $E_0$ with $E_\ell$ and $E_1$ with $E_{\ell+1}$. Then $\mathcal A^c_{\mathrm{rect}}=\sum_ke^{-i\omega\sigma_k}\Psi_{k,0}$ and

$$
\frac{d\mathcal A^c_{\mathrm{rect}}}{d\omega}=-i\sum_ke^{-i\omega\sigma_k}\bigl(\sigma_k\Psi_{k,0}+\Psi_{k,1}\bigr),\qquad \frac{d^2\mathcal A^c_{\mathrm{rect}}}{d\omega^2}=-\sum_ke^{-i\omega\sigma_k}\bigl(\sigma_k^2\Psi_{k,0}+2\sigma_k\Psi_{k,1}+\Psi_{k,2}\bigr). \tag{34}
$$

*Proof.* Differentiate $e^{-i\omega(\sigma_k+u)}$ under the integral sign: each derivative multiplies the integrand by $-i(\sigma_k+u)$. Expand $(\sigma_k+u)^2$. $\square$

The centred origin keeps every $\lvert\sigma_k\rvert$ at most $T/2$, which limits the growth of the weights in (34). The Hann derivatives follow by applying (32) to each derivative. $\mathcal A$, $\mathcal A'$ and $\mathcal A''$ are therefore all closed forms of the same rotations.

---

## 4. Dense evaluation is sampling, not the transform

Let *zoom* with density $z$ denote the evaluation of the fitted curve at $z$ equally spaced points per source interval, where $z=1$ is the source grid itself. The DFT of those dense samples, divided by $z$, is sometimes used as an approximation to the spectrum of the fitted curve. It is not the AFT.

**Proposition 6 (zoom and the AFT).** Let $h(t)=y(t)w(t)$ on the window and zero elsewhere, with $w$ the Hann weight. Let $y$ have finitely many value jumps at join points $\tau_j$ on the sampling grid, with jumps $J_j=y(\tau_j^+)-y(\tau_j^-)$. Let $z$ be a positive integer, $n$ an integer sample index and $\mu$ an integer image index. If the curve is sampled from the right at its jumps, then

$$
\frac1z\sum_nh\Bigl(t_0+\frac nz\Bigr)e^{-i\omega n/z}=\sum_{\mu\in\mathbb{Z}}\mathcal A_{\mathrm{Hann}}[y] (\omega+2\pi z\mu)+\frac1z\sum_j\tfrac12J_jw(\tau_j)e^{-i\omega(\tau_j-t_0)}. \tag{35}
$$

*Proof.* Let $f(v)=h(t_0+v)e^{-i\omega v}$ and $\hat f(\nu)=\int f(v)e^{-i\nu v}dv=\mathcal A_{\mathrm{Hann}}[y] (\omega+\nu)$. The Poisson summation formula with spacing $1/z$ [9] states $\frac1z\sum_nf(n/z)=\sum_\mu\hat f(2\pi z\mu)$, where for a piecewise smooth, compactly supported $f$ the sampled side takes the mid-value $\tfrac12\bigl(f(v^-)+f(v^+)\bigr)$ at each jump and the image sum converges symmetrically. Sampling from the right adds $\tfrac12J_jw(\tau_j)$ at each jump. The Hann weight vanishes at the window ends, so they contribute no jump. $\square$

The zoomed spectrum is therefore the AFT plus **images** at multiples of $2\pi z$, plus join terms. At $z=1$ the band above the source Nyquist frequency is a mirror of the audio band (Figure 2). By Corollary 3, the AFT of a continuous curve decays like $\omega^{-2}$, so the images near a fixed $\omega$ sum to $O(z^{-2})$. The $\omega^{-1}$ tails caused by value jumps cancel in symmetric pairs of images up to $O(z^{-2})$, while the explicit join terms of (35) are $O(z^{-1})$. As $z\to\infty$ the zoomed spectrum converges to the AFT, which FAFT computes directly.

![Figure 2](figures/faft-circle-arcs-20260930/faft-zoom-vs-transform.svg)

*Figure 2. One fitted window of a Beethoven recording (60 s), Hann applied after fitting. The exact AFT (blue) against the zoom-1 (orange) and zoom-4 (aqua) sampled transforms of the same curve, up to three times the source Nyquist frequency. Zoom 1 mirrors the band around 44.1 kHz; zoom 4 follows the AFT closely in this range. The AFT has no images.*

---

## 5. The Fast Angular Fourier Transform

### 5.1 What is computed, and what is shared

FAFT evaluates (28), or (31) for the Hann weight, at one query frequency at a time. Its elementary step multiplies a running phase by a unit complex number and adds a contribution:

$$
S\leftarrow S+\mathrm{phase}\cdot\Phi_k(\omega),\qquad \mathrm{phase}\leftarrow\mathrm{phase}\cdot q^{L_k}.
$$

This step evaluates the sum $\sum_k\Phi_k(\omega)q^{s_k-t_0}$, a generalized polynomial in $q$ whose exponents are the support starts, by accumulating powers of $q$. It is the same kind of single-frequency recurrence as Goertzel's algorithm for one DFT bin [4], applied to closed-form coefficients of a continuous curve instead of to samples. The Cooley–Tukey FFT [2] also multiplies by phases and adds, but its speed comes from a factorization of the $N\times N$ DFT matrix that shares partial sums among all $N$ bins of a fixed grid. FAFT has no such factorization and shares no work between frequencies. There is also no route through the DFT of the samples, because the AFT is not a linear function of the samples (Remark 1). The speed of FAFT comes from three properties of each rotation: its transform is a closed form (no quadrature), it needs no dense samples (no zoom), and, as shown next, it needs no transcendental function call.

### 5.2 Prepared turns, chained phases and the endpoint kernel

The exponentials in (22) factor as

$$
e^{i(\beta-\omega)L}=e^{i\beta L}q^L,\qquad e^{i(-\beta-\omega)L}=\overline{e^{i\beta L}}q^L,\qquad e^{-i\omega L}=q^L. \tag{36}
$$

The *turn* $e^{i\beta L}$ is the rotation of the radius vector over the support, $\zeta_L=\zeta_0e^{i\beta L}$. It belongs to the circle, does not depend on $\omega$, and is prepared once. For a query, one $\sin/\cos$ pair gives $q$, one power $q^L$ is formed per distinct support length (by repeated squaring, chained over the sorted lengths), and the support phases follow from (29) with one multiplication per support. The division $1/\omega$ is shared by all rotations, no transcendental function is called per rotation, and the Hann weight (31) costs three such sums. In the moment form (22), each arc costs two complex products with the turn and two real divisions, $1/(\beta-\omega)$ and $1/(-\beta-\omega)$.

*Endpoint kernel.* FAFT evaluates arcs with Proposition 2 instead. Write (14) for an arc as $y(s+u)=y_0+f(u)$, with the deviation $f(u)=a(\cos\beta u-1)+b\sin\beta u$. The deviation satisfies $f''=-\beta^2(f+a)$, so (24) with centre height $-a$, $f(0)=0$ and $q^L=e^{-i\omega L}$ gives

$$
F(\omega)=\frac{f'_Lq^L-f'_0+i\omega f_Lq^L+\beta^2aE_0(-\omega)}{(\omega-\beta)(\omega+\beta)},\qquad \Phi(\omega)=y_0E_0(-\omega)+F(\omega). \tag{37}
$$

The endpoint data $f'_0=\beta b$, $f_L=a(\cos\beta L-1)+b\sin\beta L$, $f'_L=\beta(b\cos\beta L-a\sin\beta L)$ and $\beta^2a$ do not depend on $\omega$ and are prepared once, with $\cos\beta L-1=-2\sin^2(\beta L/2)$. An arc then costs a few real products and one real division. Where (37) is a removable $0/0$, within the resonance guard $\lvert(\omega\mp\beta)L\rvert < 1/2$, FAFT evaluates the moment form (22) with the prepared turn and the series of Section 5.3. Writing $N$ and $D$ for the numerator and denominator of (37), the frequency derivatives of $F$ follow from $N=DF$:

$$
F'=\frac{N'-2\omega F}{D},\qquad F''=\frac{N''-4\omega F'-2F}{D}, \tag{38a}
$$

$$
N'=-iLq^Lf'_L+iq^Lf_L+\omega Lq^Lf_L-i\beta^2aE_1(-\omega),\qquad N''=-L^2q^Lf'_L+2Lq^Lf_L-i\omega L^2q^Lf_L-\beta^2aE_2(-\omega). \tag{38b}
$$

Since $\Psi_\ell$ is the transform of $u^\ell y$, the arc's deviation contributes $iF'$ to $\Psi_{k,1}$ and $-F''$ to $\Psi_{k,2}$ in (34), so the derivatives use the same kernel and the same guard.

### 5.3 Numerical stability

*Near DC and resonance.* When $\omega\approx0$ or $\omega\approx\pm\beta$, the ratio (19) divides two small numbers. Let $\chi=\delta L$ be the accumulated phase offset over the support and $\varepsilon=2^{-52}$ the machine epsilon of double precision. For $\lvert\chi\rvert$ below a threshold, $0.05$ for values and $0.5$ for derivatives, the moments are evaluated from their power series

$$
E_\ell(\delta,L)=L^{\ell+1}\sum_{\nu\ge0}\frac{(i\chi)^\nu}{\nu!(\nu+\ell+1)}, \tag{39}
$$

with $\nu$ a nonnegative integer term index. The even and odd powers of $i\chi$ give the real and imaginary parts, each evaluated by Horner's rule in $\chi^2$ with precomputed coefficients; 8 to 18 terms keep the truncation error at the level of the rounding error. Above the threshold, cancellation gives the ratio (19) and its recurrence a relative error of at most about $\varepsilon/\lvert\chi\rvert^{\ell+1}$ [10]: below $10^{-13}$ for values and $10^{-11}$ for derivatives.

*Chained phases.* Each multiplication in (29) adds a relative rounding error of order $\varepsilon$, so the error of the running phase grows with the number of supports chained [10]. The chain is re-anchored with an exact $\sin/\cos$ every 64 supports.

*Flat arcs.* An arc with a large radius has a centre far away: its centre height $c$ is a large number of sign opposite to $r\sin\varphi$, and the two nearly cancel in $y_0=c+a$. The anchored forms (14) and (22) start from the evaluated start value $y_0$ and never form $c$ again. In the moment form (22), however, the $a$ and $b$ terms still carry the radius, and the bracketed differences of moments are computed with an absolute error of order $\varepsilon L$, so one rotation's rounding error is of order $\varepsilon rL$. The endpoint kernel (37) combines only quantities of the size of the deviation itself: its end change, its slopes and its curvature $\beta^2a$, all prepared with relative accuracy. Away from resonance it therefore computes the deviation of a flat arc with relative accuracy. For an arc with $r=10^8$ and a height near 50, the implementation's tests require agreement with quadrature of the evaluated curve to $10^{-12}$ relative; the moment form misses this by two orders of magnitude. Near $\omega=\pm\beta$, numerator and denominator of (37) vanish together. The relative error then grows like $\varepsilon/\lvert\chi\rvert$, and like $6\varepsilon/\chi^2$ for its curvature term, with $\chi=(\omega\mp\beta)L$; this is why the guard of Section 5.2 hands those arcs to the moment form. The fitter of Section 7 also does not accept circles whose radius exceeds $10^6$ times the peak amplitude of the frame; such supports use a line or are split.

### 5.4 Algorithms

In the pseudocode, $o$ is the crop offset from the original rotation start, $s'$ and $L'$ are the cropped start and duration, `turn` is the prepared rotation $e^{i\beta L'}$ of the radius vector over the cropped support, and $f'_0$, $f_L$, $f'_L$, $\kappa=\beta^2a$ are the endpoint data of (37). In Algorithm 2, $Q[L]$ stores $q^L$, $S$ accumulates the transform, `phase` is the chained support phase of (29), and $e_{\mathrm{dc}}$, $e_+$, $e_-$ are the moments $E_0$ at $-\omega$, $\beta-\omega$ and $-\beta-\omega$. `conj` denotes complex conjugation, $R$ the number of rotations in the prepared window and $K$ the number of queried frequencies.

```text
Algorithm 1  PREPARE(curve, window [t0, t0+T), weighting)
  for each rotation (s, L, c, r, φ, β, d) overlapping the window:
      crop to [max(s,t0), min(s+L,t0+T));   o ← max(s,t0) − s
      y0 ← y(s+o);  a ← r·sin(φ+βo);  b ← r·cos(φ+βo)    ▷ anchored form (14)
      turn ← e^{iβL'}                                      ▷ independent of ω
      f'0 ← βb;  f_L ← a(cos βL' − 1) + b·sin βL'          ▷ endpoint data (37),
      f'_L ← β(b·cos βL' − a·sin βL');  κ ← β²a            ▷ cos−1 = −2sin²(βL'/2)
      store (s'−t0, L', y0, a, b, β, d, turn, f'0, f_L, f'_L, κ)
  mark each rotation whose successor is stacked on the same support
  lengths ← sorted distinct L'
  return prepared window
```

```text
Algorithm 2  RECT(prepared, ω)                       ▷ Proposition 3, O(R)
  q ← e^{−iω};  Q[L] ← q^L for L in lengths          ▷ one sin/cos, few products
  S ← 0;  phase ← 1
  for each rotation k in time order:
      e_dc ← E0(−ω, L_k; Q[L_k])
      Φ ← y0·e_dc                                    ▷ Corollary 1
      if arc and |(ω−β)L_k| ≥ ½ and |(ω+β)L_k| ≥ ½:     ▷ endpoint kernel (37)
               Φ ← Φ + (f'_L·Q[L_k] − f'0 + iω·f_L·Q[L_k] + κ·e_dc) / ((ω−β)(ω+β))
      else if arc:                                   ▷ moment form (22)
               e₊ ← E0(β−ω; turn·Q[L_k]);   e₋ ← E0(−β−ω; conj(turn)·Q[L_k])
               Φ ← Φ + a·((e₊+e₋)/2 − e_dc) + b·(e₊−e₋)/(2i)
      if d≠0:  Φ ← Φ + d·E1(−ω; Q[L_k], e_dc)
      S ← S + phase·Φ                                ▷ rotate and add
      if the next rotation is not stacked on this support:
          phase ← phase·Q[L_k]                       ▷ (29); re-anchor every 64 steps
  return S

Algorithm 3  AFT(prepared, ω)                        ▷ normalized coefficient
  if rectangular: return RECT(ω)/T
  Δ ← 2π/T
  return (½RECT(ω) − ¼RECT(ω−Δ) − ¼RECT(ω+Δ)) / (T/2)  ▷ Proposition 4
```

**Complexity.** Preparation is $O(R)$ for $R$ rotations in the window. One frequency costs $O(R)$, independent of how densely the curve could be zoomed. $K$ arbitrary frequencies cost $O(RK)$; for a dense grid with $K\sim N$ and $R\propto N$, this is quadratic. FAFT is not an alternative to the FFT for computing all DFT bins of the samples, which is a different quantity.

---

## 6. Relevant frequencies on a continuous axis

The AFT is a smooth function of $\omega$. Its relevant frequencies form a variable-length list of real numbers, not $N/2+1$ bins. Two questions must be kept apart. The first is where the local maxima of $\lvert\mathcal A\rvert$ lie: a property of the transform, answered exactly by Newton's method (Section 6.1). The second is which components explain the curve and how strong each one is: a question about a *model* of the curve, because the height of a maximum is shared with its neighbours through the window (Sections 6.2 and 6.3). Sections 6.4 to 6.6 answer the second question with a joint fit whose every term is a closed form, and Section 6.7 carries the selection from one window to the next.

### 6.1 Maxima of the transform

The resolution of the transform is set by the window *duration*, through $2\pi/T$, not by the number of samples. A zoomed curve has the same $T$ and therefore the same resolution. Seeds are taken on a grid of spacing $2\pi/(T\rho)$, where the density $\rho$, a positive integer with default 2, is the number of seed spacings per interval of width $2\pi/T$. Each seed is refined by Newton's method on the spectral power $G(\omega)=\lvert\mathcal A(\omega)\rvert^2$, using Proposition 5. With $\mathrm{Re}$ the real part,

$$
G'=2\mathrm{Re}\bigl(\overline{\mathcal A}\mathcal A'\bigr),\qquad G''=2\bigl(\lvert\mathcal A'\rvert^2+\mathrm{Re}\bigl(\overline{\mathcal A}\mathcal A''\bigr)\bigr),\qquad \omega\leftarrow\omega-\frac{G'}{G''}\quad\bigl(G'' < 0\bigr). \tag{40}
$$

The Newton step is safeguarded by a bracket that shrinks with the sign of $G'$ and falls back to bisection, as in standard safeguarded root finding [11]. In Algorithm 4, $\omega_{\mathrm{lo}}$ and $\omega_{\mathrm{hi}}$ bound the search band, $j$ indexes scan frequencies, `floor` is the minimum accepted magnitude, and `lo` and `hi` bracket a candidate peak. $A$, $A'$, $A''$ are the transform and its frequency derivatives, $G_1$, $G_2$ the first and second derivatives of the power, and $\omega_{\mathrm{new}}$ the proposed next query frequency.

```text
Algorithm 4  PEAKS(prepared, [ω_lo, ω_hi], ρ, floor)
  scan |AFT| on ω_j = ω_lo + j·2π/(Tρ)          ▷ Hann reuses one RECT grid
  for each local maximum j above floor:
      ω ← parabolic vertex of log|AFT| at j−1, j, j+1
      [lo, hi] ← [ω_{j−1}, ω_{j+1}]
      repeat                                     ▷ safeguarded Newton, (40)
          (A, A', A'') ← DERIVATIVES(ω)          ▷ Proposition 5
          G1 ← 2Re(conj(A)A');  G2 ← 2(|A'|² + Re(conj(A)A''))
          if G1 > 0: lo ← ω;  if G1 < 0: hi ← ω
          ω_new ← ω − G1/G2 if G2 < 0 and inside [lo,hi] else (lo+hi)/2
      until |ω_new − ω| ≤ 10⁻¹²
      drop if pinned to a bracket edge (a flank) or not converged
  merge peaks closer than π/(2T), keeping the strongest; apply floors
  return peaks (ω, A(ω))
```

The grouping, merging and exclusion steps are explicit. Nearby candidates are merged, weak ones are excluded by relative and absolute floors, and flanks and unconverged seeds are discarded and counted. Every reported frequency is an exact maximum of $\lvert\mathcal A\rvert$. The scan is a seed heuristic, not a completeness proof, and finer query precision does not separate two maxima closer than about $2\pi/T$: an arbitrary query position is not added resolution. Algorithm 4 reports where the transform peaks; the following sections show why that is not yet a list of components.

### 6.2 The window kernel and the range of a peak

Let $t_c=t_0+T/2$ be the window centre. The *kernel* of a weighting is the transform of the weight itself, with phase origin at the centre:

$$
K(\nu)=\int_{t_0}^{t_0+T}w(t)e^{-i\nu(t-t_c)}dt. \tag{41}
$$

Both weightings are symmetric about $t_c$, so $K$ is real and even, $K(0)=W$, and its odd derivatives vanish at $0$. For the rectangular weight, and by Proposition 4 for Hann, with $\Delta=2\pi/T$,

$$
K_{\mathrm{rect}}(\nu)=\frac{2\sin(\nu T/2)}{\nu}=T\operatorname{sinc}\frac{\nu T}{2},\qquad K_{\mathrm{Hann}}(\nu)=\tfrac12K_{\mathrm{rect}}(\nu)+\tfrac14K_{\mathrm{rect}}(\nu-\Delta)+\tfrac14K_{\mathrm{rect}}(\nu+\Delta). \tag{42}
$$

**Proposition 7 (tone response).** Let a real tone of frequency $\omega_0$ be $x(t)=\alpha\cos\bigl(\omega_0(t-t_c)\bigr)+\beta\sin\bigl(\omega_0(t-t_c)\bigr)$ and $c=(\alpha-i\beta)/2$. Its centred transform is, at every real $\omega$,

$$
\mathcal A^c[x] (\omega)=c K(\omega-\omega_0)+\overline cK(\omega+\omega_0). \tag{43}
$$

*Proof.* Write $x=c e^{i\omega_0(t-t_c)}+\overline ce^{-i\omega_0(t-t_c)}$ and apply (41) to each exponential. $\square$

A tone therefore occupies the shape of $K$ around $\omega_0$, plus an image at $-\omega_0$, and nothing else: there is no frequency grid and there are no spectral images at multiples of $2\pi$ (compare Proposition 6). The normalized height at the tone is $\lvert c\rvert\cdot K(0)/W=\lvert c\rvert$, half the amplitude. Table 1 lists the shape of $K$ in units of the resolution $2\pi/T$ ("bins"); these are the classical window figures [6], here exact functions of a continuous variable.

| Kernel, in bins of $2\pi/T$ | Rectangular | Hann |
|---|---:|---:|
| Half-width at −3 dB / −6 dB / −20 dB | 0.442 / 0.603 / 0.908 | 0.719 / 0.998 / 1.650 |
| First null (end of the main lobe) | 1 | 2 |
| Highest sidelobe | −13.26 dB at 1.43 | −31.47 dB at 2.36 |
| Sidelobe decay | 6 dB per octave | 18 dB per octave |
| Height loss of a tone half a bin from the DFT grid | 3.92 dB | 1.42 dB |
| Height loss on the continuous axis | 0 | 0 |

*Table 1. Shape of the window kernel. The last two rows compare the "scalloping" of a fixed DFT grid with the AFT, whose maxima are located on the continuous axis.*

The continuous axis removes grid scalloping. It does not remove *leakage*: by (43) and linearity, the transform of a sum of tones at a selected frequency $\omega_p$ is

$$
\mathcal A^c(\omega_p)=c_pK(0)+\overline{c_p}K(2\omega_p)+\sum_{q\ne p}\Bigl[c_qK(\omega_p-\omega_q)+\overline{c_q}K(\omega_p+\omega_q)\Bigr]+\mathcal A^c[\mathrm{rest}] (\omega_p), \tag{44}
$$

where "rest" is the non-tonal part of the curve, including the join terms of Corollary 3. The decibel level of a selected frequency therefore involves its neighbours: a neighbour at a separation of $s$ bins with relative level $L$ can change the height by up to $20\log_{10}\bigl(1\pm10^{L/20}\max_{s'\ge s}\lvert K(s')\rvert/K(0)\bigr)$ dB, depending on the relative phase. Table 2 evaluates this envelope.

| Neighbour at $\ge s$ bins | Rect, equal level | Rect, −20 dB | Rect, −40 dB | Hann, equal level | Hann, −20 dB | Hann, −40 dB |
|---|---:|---:|---:|---:|---:|---:|
| $s=1$ | +1.71 / −2.13 | ±0.19 | ±0.019 | +3.52 / −6.02 | ±0.42 | ±0.043 |
| $s=2$ | +1.05 / −1.19 | ±0.11 | ±0.011 | +0.23 / −0.24 | ±0.023 | ±0.002 |
| $s=4$ | +0.60 / −0.64 | ±0.06 | ±0.006 | +0.03 / −0.03 | ±0.003 | ±0.0003 |
| $s=8$ | +0.32 / −0.33 | ±0.03 | ±0.003 | < 0.01 | < 0.001 | < 0.0001 |

*Table 2. Largest change, in dB, of a peak's height caused by one neighbouring tone at a separation of at least $s$ bins, by its level relative to the peak. The rectangular kernel trades a narrower main lobe for slowly decaying sidelobes; the Hann kernel confines leakage to about two bins.*

**Peak range.** The *range* of a peak at $\omega_p$ at depth $D$ dB is the interval around $\omega_p$ where $\lvert\mathcal A\rvert\ge\lvert\mathcal A(\omega_p)\rvert10^{-D/20}$. It is measured on the transform itself: from $\omega_p$ outwards in steps of $2\pi/(16T)$, then by bisection to the crossing. A side that meets a valley above the level stops at the valley's minimum and is flagged as *shared*: the neighbouring maximum owns part of that range. For an isolated tone the range equals the kernel's half-width of Table 1 (tested to $10^{-3}$ of a bin). A range wider than the kernel, or a shared side, indicates unresolved neighbours, a modulated (non-stationary) component, or non-tonal content.

The range depends on how the transform is obtained, in four ways. (i) The *weighting* sets $K$ (Table 1). (ii) A *sampled* evaluation of the same curve at zoom $z$ adds the images of Proposition 6 at multiples of $2\pi z$, and the DFT of the samples adds a periodic Dirichlet kernel and grid scalloping; the AFT has neither. (iii) The *moment and endpoint kernels* of Section 5.2 compute the same numbers (to $5\cdot10^{-15}$) and do not change ranges. (iv) The *curve* changes the non-tonal part of (44): by Corollary 3, kinks and value jumps at joins add a broadband floor decaying like $\omega^{-2}$ and $\omega^{-1}$, so a hierarchical fit with a coarser tolerance, or the triplet curve with a kink at every sample, lowers the dynamic range within which ranges and weak peaks are meaningful.

### 6.3 Why peak picking is a zero-sum selection

Equation (44) makes the difficulty of Section 6.1 precise. Reading a list of maxima as a list of components has three consequences.

1. *Biased heights and positions.* Each reported height contains the leakage of every other component, up to the amounts of Table 2, and the maximum itself is shifted towards or away from its neighbours.
2. *Spurious maxima.* Sidelobes are local maxima. With the rectangular weight a single tone has sidelobes at −13 dB, −18 dB, −21 dB and so on, all above a 1% floor (−40 dB); Algorithm 4 reports them as peaks.
3. *Hidden components.* A weak tone on the flank or sidelobe of a strong one has no maximum of its own, or only a biased one.

What is not selected is not discarded: its energy stays in the heights of the selected peaks and in spurious maxima. Every missed component makes the others less accurate, and a component split into two maxima takes height from itself. This is the same mechanism by which the energy of an off-bin tone is shared among fixed DFT bins, except that the selected frequencies are free. Choosing maxima more finely (a denser scan) cannot remove it, because it is a property of $K$ and $T$, not of the search. The selection must instead be made jointly: all selected components are estimated together, each height with its neighbours' leakage removed, and the energy that remains unexplained is measured.

### 6.4 Joint tone fit in closed form

Let the selected frequencies be $\omega_1,\ldots,\omega_P$, write $u=t-t_c$, and take the tone atoms $C_p=\cos\omega_pu$ and $S_p=\sin\omega_pu$. We fit $\sum_p\alpha_pC_p+\beta_pS_p$ to the curve by weighted least squares with the inner product $\langle f,g\rangle=W^{-1}\int f g w dt$. Let $k=K/W$ be the normalized kernel and $a=\mathcal A^c/W$ the normalized centred transform. Products of cosines and sines reduce to (41), and the cross terms vanish because $w$ is symmetric:

$$
\langle C_p,C_q\rangle=\frac{k(\omega_p-\omega_q)+k(\omega_p+\omega_q)}2,\qquad \langle S_p,S_q\rangle=\frac{k(\omega_p-\omega_q)-k(\omega_p+\omega_q)}2,\qquad \langle C_p,S_q\rangle=0. \tag{45}
$$

The data side consists of transform values at the selected frequencies:

$$
\langle y,C_p\rangle=\operatorname{Re}a(\omega_p),\qquad \langle y,S_p\rangle=-\operatorname{Im}a(\omega_p). \tag{46}
$$

The normal equations split into a cosine system $G_c\alpha=b_c$ and a sine system $G_s\beta=b_s$ of size $P$, with the matrices (45) and right sides (46), solved by Cholesky factorization. Each fitted component has the normalized value $c_p=(\alpha_p-i\beta_p)/2$, comparable with $\lvert\mathcal A(\omega_p)\rvert/W$; the ratio of the two, in dB, is the leakage of (44) at that peak.

**Proposition 8 (residual energy and certificate).** Let $E=\int y^2w dt$ be the weighted energy of the curve, and let $R(\omega)=\mathcal A^c(\omega)-W\sum_p\bigl[c_pk(\omega-\omega_p)+\overline{c_p}k(\omega+\omega_p)\bigr]$ be the transform of the residual curve $r=y-\sum_p(\alpha_pC_p+\beta_pS_p)$. Then

$$
E_r=\int r^2w dt=E-W\bigl(\alpha^{\mathsf T}b_c+\beta^{\mathsf T}b_s\bigr), \tag{47}
$$

$E_r$ does not increase when a frequency is added to the list, and for every real $\omega$

$$
\frac{\lvert R(\omega)\rvert}W\le\sqrt{\frac{E_r}W}. \tag{48}
$$

*Proof.* (47) is the residual of a least-squares fit, by orthogonality of the residual to the atoms. Adding atoms enlarges the subspace, so the minimum cannot grow. For (48), $\lvert\int rwe^{-i\omega u}dt\rvert=\lvert\int(r\sqrt w)(\sqrt we^{-i\omega u})dt\rvert\le\sqrt{\int r^2w dt}\sqrt{\int w dt}$ by the Cauchy–Schwarz inequality, since $w\ge0$. $\square$

Proposition 8 turns the zero-sum effect into an account. $E_r/E$ is the fraction of the curve left unexplained by the selection, and the right side of (48) is a *certificate*: wherever it is below the floor, no component above the floor is missing anywhere on the real axis, including above the source Nyquist frequency. The bound is conservative when the residual is broadband, because it allows all of $E_r$ to sit at one frequency. $E$ is the only quantity of this section computed by numerical integration: Gauss–Legendre rules with 16 nodes on panels of each support, with at most $\pi$ of phase per panel (twice the rate plus the Hann rate), which is exact to rounding. It is one scalar per window, not a transform value.

The *significance* of component $p$ is its unique contribution: the increase of $E_r$ when $p$ is removed and the others refitted, expressed as the height of an isolated tone with the same energy. By the Schur complement of the normal equations,

$$
\sigma_p=\sqrt{\frac{\Delta E_p}{2W}},\qquad \Delta E_p=W\Bigl[\frac{\alpha_p^2}{(G_c^{-1})_{pp}}+\frac{\beta_p^2}{(G_s^{-1})_{pp}}\Bigr]. \tag{49}
$$

For an isolated tone $\sigma_p=\lvert c_p\rvert$. When two components share energy, for example two fits of one tone or two tones within one main lobe, $\sigma_p<\lvert c_p\rvert$. Components are pruned by significance, the least significant first, which removes duplicates and sidelobe fits without imposing a minimum distance between distinct tones.

*Seed frequency.* A new component is placed at the maximum of the energy that one real tone explains in the current residual $r=R/W$,

$$
P(\omega)=\frac{\bigl(\operatorname{Re}r(\omega)\bigr)^2}{g_c(\omega)}+\frac{\bigl(\operatorname{Im}r(\omega)\bigr)^2}{g_s(\omega)},\qquad g_{c,s}(\omega)=\frac{1\pm k(2\omega)}2, \tag{50}
$$

which is the least-squares frequency of a single real tone. For a complex tone it reduces to the maximum of $\lvert r\rvert^2$, the classical periodogram estimator [12]; the denominators account for the tone's own image at $-\omega$, which biases the maximum of $\lvert r\rvert$ with the rectangular kernel. Newton on (50) uses the derivatives of $r$ (Proposition 5 minus closed-form kernel derivatives) and of $k$. Within one bin of $\omega=0$, where the sine atom vanishes, $\lvert r\rvert^2$ is used.

### 6.5 Joint refinement of frequencies

Refining each frequency on its partial residual with the others fixed, a coordinate method such as RELAX [13], converges slowly when components are strongly coupled. In a development comparison on exact tones 1–2 bins apart (four relative phases each), up to four such sweeps per round left a spurious third component 15–25 dB below the stronger tone in 10 of 16 configurations, and heights biased by up to 2.4 dB in 2 more. FAFT instead refines all frequencies together by Gauss–Newton on the full nonlinear least-squares problem, with Levenberg–Marquardt damping [14, 15], and re-solves the amplitudes exactly at each new set of frequencies (variable projection [16]). The derivative of a tone with respect to its frequency is the atom $D_p=u(-\alpha_pS_p+\beta_pC_p)$. Its inner products follow from (41) by differentiation under the integral, with $k'$ and $k''$ the derivatives of $k$:

$$
\langle C_p,D_q\rangle=\frac{\alpha_q}2\bigl[k'(\omega_p+\omega_q)-k'(\omega_p-\omega_q)\bigr],\qquad \langle S_p,D_q\rangle=-\frac{\beta_q}2\bigl[k'(\omega_p-\omega_q)+k'(\omega_p+\omega_q)\bigr], \tag{51}
$$

$$
\langle D_p,D_q\rangle=\frac{\alpha_p\alpha_q}2\bigl[k''(\omega_p+\omega_q)-k''(\omega_p-\omega_q)\bigr]-\frac{\beta_p\beta_q}2\bigl[k''(\omega_p-\omega_q)+k''(\omega_p+\omega_q)\bigr], \tag{52}
$$

and the data side needs the transform and its first derivative at each selected frequency, which Proposition 5 provides in closed form:

$$
\langle y,D_p\rangle=\alpha_p\operatorname{Re}a'(\omega_p)-\beta_p\operatorname{Im}a'(\omega_p). \tag{53}
$$

The right side of the Gauss–Newton system is $\langle y,\cdot\rangle$ minus the model's inner products, obtained from (45) and (51) with the current amplitudes. A step is kept if $E_r$ does not grow beyond its own rounding ($64\varepsilon E$), and the damping then shrinks by 4; otherwise it grows by 8. Steps are limited to half a bin. Components farther apart than 4 bins (Hann; 8 for the rectangular kernel) are solved as separate clusters, each with its own damping, because their coupling is below −48 dB (Table 2); the global model still enters every right side. A cluster stops when every proposed step is below a stated tolerance, by default $10^{-4}$ bin, without evaluating that step. One iteration costs one evaluation of $(\mathcal A,\mathcal A')$ per moving component plus $O(m^3)$ algebra for a cluster of $m$ components; no scan, no sampling and no quadrature are involved.

*Resolution.* On exact tones, the joint fit separates two tones 1.0 and 1.5 bins apart at every tested relative phase, to $10^{-6}$ of a bin, although $\lvert\mathcal A\rvert$ has a single maximum for the pair. This is not resolution added by the transform, which still has the kernels of Table 1. It is the information contained in the assumption that the data are tones. For noisy, modulated or non-tonal content no such separation is guaranteed, and the significance (49) decides whether a second component is supported by the energy it uniquely explains.

### 6.6 The pursuit algorithm

Algorithm 5 combines these steps. The transform is scanned once, as in Algorithm 4. Later rounds seed from local maxima of the *residual* on the same grid, $\bigl\lvert a(\omega_j)-\sum_p\bigl[c_pk(\omega_j-\omega_p)+\overline{c_p}k(\omega_j+\omega_p)\bigr]\bigr\rvert$. This costs $O(P)$ kernel evaluations per grid point and no new transform values. Within a round, the components accepted first carry provisional amplitudes, so a seed they already explain, typically a sidelobe, is skipped before any Newton step. This is the idea of the CLEAN deconvolution of radio astronomy [17] and of matching pursuit [18], with an orthogonal (joint) refit after each round as in orthogonal matching pursuit [19]. In Algorithm 5, $Z_j$ are the scanned centred values, $\mathcal C$ the component list, and $\mathcal X$ the rejected seed positions.

```text
Algorithm 5  PURSUE(prepared, [ω_lo, ω_hi], ρ, floor)
  E ← ENERGY(prepared)                                   ▷ Proposition 8
  Z_j ← AFT^c(ω_j) on the scan of Algorithm 4             ▷ the only scan
  floor ← max(absolute, relative · max_j |Z_j|);  C ← ∅;  X ← ∅
  repeat for at most 6 rounds:
      r_j ← Z_j − Σ_{p∈C} [c_p k(ω_j−ω_p) + conj(c_p) k(ω_j+ω_p)]
      for each local maximum of |r_j| ≥ floor, strongest first, not near C ∪ X:
          if components added in this round explain it below floor: skip
          ω ← Newton maximum of P(ω), (50), on the residual
          add ω to C with its provisional amplitude
      if nothing was added: stop (no seeds)
      POLISH(C)                                           ▷ Section 6.5, (51)–(53)
      while the least significant σ_p < floor:             ▷ (49)
          move p to X; refit; neighbours of p become active
      if any were removed: POLISH(C) and prune again
      if √(E_r/W) ≤ floor: stop (certified)               ▷ (48)
  POLISH(C) once more with every component active; prune
  return C with ω_p, c_p, A(ω_p), σ_p, and E, E_r, √(E_r/W)
```

The final pass re-checks components settled in early rounds, which still carried the weak leakage of clusters refined later. Every reported frequency is a converged minimum of the joint residual within the stated tolerance, and every reported value is either a fitted amplitude or an exact transform value. The stopping reason is reported: no seed above the floor, a certificate below the floor, or a round or capacity limit.

### 6.7 Tracking consecutive windows

When windows of one input are analysed in order, a frame's components predict the next frame's: the frequency of a stable partial changes little over a hop, and its amplitude is re-solved in the new window (previous phases belong to another window). Algorithm 6 therefore replaces the scan by a joint polish started at the predicted frequencies, and pays for a full pursuit only when needed. Peak tracking with births and deaths of partials is standard in sinusoidal modelling [20, 21]; what is specific here is that a tracked frame needs no scan and carries the certificate (48). Track identities give the selection stability; no smoothing filter is applied to the reported values, which remain exact optima of each window's fit.

A tone $\alpha C+\beta S$ of normalized height $m=\lvert c\rvert$ that is absent from the model, and uncorrelated with it, adds its own weighted energy to the residual:

$$
\Delta E_r=W\Bigl[2m^2+\tfrac12\bigl(\alpha^2-\beta^2\bigr)k(2\omega)\Bigr]\approx2m^2W, \tag{54}
$$

where the image term $k(2\omega)$ is negligible away from $\omega=0$. A new component above the birth floor $m_b$ therefore cannot appear without either breaking the certificate (48) or raising $E_r$ by about $2m_b^2W$. Algorithm 6 rescans in that case. It also rescans at a fixed interval, because a birth that coincides with an equal fall of residual energy elsewhere is not detected by (54); this novelty test is a heuristic.

```text
Algorithm 6  TRACK(next prepared window, tracks)
  E ← ENERGY;  jump ← 2π/T;  every track predicts ω̂ = ω + clamp(ω − ω_prev, ±jump/2)
  if no tracks, or the rescan interval is due: go to SCAN
  C ← tracks at ω̂;  POLISH(C)                       ▷ no scan
  drop components that moved more than jump from ω̂, then prune by σ_p
  floor ← relative · max |A(ω_p)|;  b ← √(E_r/W)
  if b ≤ floor (certified) or E_r ≤ E_ref + 2 m_b² W:   ▷ (48), (54)
      continue matched tracks;  E_ref ← min(E_ref, E_r);  return
SCAN:
  C ← PURSUE;  E_ref ← E_r
  match components to predictions within jump, strongest first
  unmatched component with |c| ≥ birth floor: new track     ▷ hysteresis:
  unmatched track: missed; ended after `hold` missed frames  ▷ birth > survival
```

In the synthetic test of Section 7, a tracked window cost two to four evaluations of $(\mathcal A,\mathcal A')$ per component, instead of the scan of Algorithm 4, which needs $\rho(\omega_{\mathrm{hi}}-\omega_{\mathrm{lo}})T/(2\pi)$ transform values. It never reports a frequency that is not a converged optimum of the current window.

---

## 7. Experiments

**Setup.** A Beethoven recording (44.1 kHz) was decoded to PCM16 and the channels averaged to mono. Excerpts of two seconds at 60, 180 and 300 s were used, each contributing 21 evenly spaced windows of $N=1024$ samples. The circles were fitted hierarchically on variable supports of at most 128 samples, with every source sample within 16 PCM16 units. The Hann window was applied after fitting. Timings are warmed medians on an arm64 laptop, Rust 1.98.1.

**Exactness.** The AFT was compared with an independent Gauss–Legendre quadrature of the exported rotations; the largest relative difference was $1.8\cdot10^{-14}$. All 1,219 reported peaks were confirmed as local maxima by the same quadrature. On the triplet curve of Section 2.7, FAFT matched an independent per-interval evaluation of the same integral (closed-form sinc integrals with trigonometric calls on every interval) to $6.5\cdot10^{-14}$.

**Zoom convergence (Proposition 6).** The median relative $L^2$ difference between the zoom-$z$ sampled transform and the AFT, below the Nyquist frequency, is:

| $z$ | 1 | 2 | 4 | 8 | 16 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| 60 s | $2.1\cdot10^{-3}$ | $4.6\cdot10^{-4}$ | $1.2\cdot10^{-4}$ | $3.9\cdot10^{-5}$ | $1.5\cdot10^{-5}$ | $3.3\cdot10^{-6}$ |
| 180 s | $1.3\cdot10^{-3}$ | $2.9\cdot10^{-4}$ | $7.5\cdot10^{-5}$ | $2.1\cdot10^{-5}$ | $6.9\cdot10^{-6}$ | $1.6\cdot10^{-6}$ |
| 300 s | $1.8\cdot10^{-3}$ | $4.2\cdot10^{-4}$ | $1.1\cdot10^{-4}$ | $3.7\cdot10^{-5}$ | $1.4\cdot10^{-5}$ | $3.0\cdot10^{-6}$ |

The decay is close to $z^{-2}$ at low zoom (images) and approaches $z^{-1}$ at high zoom (join terms), as Proposition 6 and Corollary 3 predict.

**Cost.** Per-window medians at $N=1024$:

| | 60 s | 180 s | 300 s |
|---|---:|---:|---:|
| Rotations per window | 247 | 141 | 546 |
| FAFT, one Hann frequency | 4.17 µs | 2.44 µs | 10.61 µs |
| Sampled-curve kernel, rotations (zoom 1) | 7.99 µs | 5.57 µs | 14.43 µs |
| Sampled-curve kernel, closed sums (zoom 1) | 54.22 µs | 34.31 µs | 95.01 µs |
| Per-interval AFT (trigonometric calls per interval), triplet curve | 41.00 µs | 41.06 µs | 41.12 µs |
| FAFT, same triplet curve | 12.22 µs | 11.91 µs | 12.42 µs |
| FAFT relevant peaks (scan + Newton) | 3.10 ms | 1.42 ms | 10.93 ms |
| Adaptive sampled-curve peak search | 27.42 ms | 16.91 ms | 49.44 ms |
| FFT of the source samples (plain / 64× padded) | 5.8 / 280 µs | 5.8 / 280 µs | 5.9 / 281 µs |

Query cost grows linearly with the number of rotations; the sizes 256…4096 gave 0.97, 1.69, 3.51, 7.09 and 15.47 µs per frequency (Figure 3).

**Kernel.** The Beethoven measurements in this section used the moment form (22) for every arc and were not repeated. The endpoint kernel (37) was compared with it afterwards on three synthetic 1,024-sample frames (three tones plus noise, hierarchical fits with source tolerances 4, 16 and 64, 644–962 rotations per window), by building both versions of the implementation and alternating them over 256 frequencies. One query became 1.11–1.12× faster with the rectangular weight and 1.18–1.20× faster with Hann, a Hann peak list 1.15× faster, and a query on the triplet curve 1.07× faster; with unit-length arcs, the resonance guard covers a wider band of frequencies. Values changed by at most $5\cdot10^{-15}$ relative to the largest value, and the peak lists were identical.

![Figure 3](figures/faft-circle-arcs-20260930/faft-convergence-cost.svg)

*Figure 3. Left: difference between zoom-sampled transforms and the AFT. Middle: cost of one arbitrary frequency against window size. Right: cost of a list of relevant peaks, with the FFT of the samples as a reference for a different target.*

**Continuous peaks.** The peaks in this paragraph are the maxima of Algorithm 4. The FAFT returned on average 19.2, 13.1 and 25.7 peaks per window. Every one was also found by the adaptive sampled-curve search, which found about 5% more. Those extra peaks are weak maxima next to stronger ones and are missed by the density-2 scan; density 4 recovers most of them. Of the peaks of the 64×-padded FFT of the source samples, 89–94% were recovered, with a median distance of 0.18–0.44 Hz. The bin spacing is 43 Hz; the reported AFT peaks are converged to $10^{-12}$ rad/sample on the continuous axis (Figure 4).

![Figure 4](figures/faft-circle-arcs-20260930/faft-peaks.svg)

*Figure 4. The AFT of one window (blue) with its Newton-refined peaks, against the 64×-padded FFT of the source samples (orange). The ticks at the bottom are the fixed $N=1024$ DFT bins; the AFT peaks fall between them.*

**Peak selection (synthetic).** Algorithms 4, 5 and 6 were compared on a signal with known partials, generated deterministically by the reference implementation: 44.1 kHz, 48 Hann windows of $N=1024$ samples with a hop of 512. It contains a harmonic note gliding from 220 to 226 Hz (8 partials, amplitudes $1000/k$), a second harmonic note at 330.4 Hz (5 partials, $400/k$) that enters halfway, a pair of tones at 2400 Hz and 1.5 bins above it (amplitudes 300 and 100), a tone of amplitude 20 at 3150 Hz (−40 dB relative to the strongest), and uniform noise of ±3. The circles were fitted hierarchically with a source tolerance of 4 and supports of at most 128 samples (547 rotations per window). All three algorithms used a relative floor of 1% (−40 dB). The truth is the instantaneous frequency of each partial at the window centre; the two windows that straddle the onset are not scored (46 windows, 13.5 partials per window). A partial counts as recovered when a reported frequency lies within half a bin, and a report farther than one bin from every partial counts as spurious. Heights compare the reported magnitude with $A/2$: Algorithm 4 reports $\lvert\mathcal A(\omega_p)\rvert/W$, Algorithms 5 and 6 the fitted $\lvert c_p\rvert$. Times are medians of five warmed runs on the same laptop.

| | Algorithm 4 (scan and Newton) | Algorithm 5 (pursuit) | Algorithm 6 (tracking) |
|---|---:|---:|---:|
| Partials recovered | 89.2% | 100% | 100% |
| Weak partials recovered (the −40 dB tone and the weaker tone of the pair) | 63.0% | 100% | 100% |
| Reported per window / spurious per window | 12.7 / 1.33 | 13.4 / 0 | 13.5 / 0 |
| Height error, median / 95th percentile | 0.044 / 4.70 dB | 0.009 / 0.56 dB | 0.009 / 0.58 dB |
| Frequency error, median / 95th percentile, in bins | $6.2\cdot10^{-3}$ / 0.30 | $5.2\cdot10^{-4}$ / 0.017 | $5.3\cdot10^{-4}$ / 0.017 |
| Evaluations of $(\mathcal A,\mathcal A',\mathcal A'')$ / rectangular sums, per window | 42 / 1,029 | 200 / 1,029 | 70 / 134 |
| Time per window, median (mean) | 6.4 (6.3) ms | 20.6 (20.4) ms | 3.5 (6.7) ms |

*Table 3. Peak selection on a synthetic signal with known partials (aggregate snapshot `docs/experiments/faft-pursuit-synthetic-20261004.json`).*

With the rectangular weight on the same windows, Algorithm 4 reported 55.2 maxima per window, most of them sidelobes, and Algorithm 5 reported 14.5 components. The joint fit explained a median 99.9992% of the weighted energy, and its certificate (48) was a median 0.51 of the floor. The pursuit costs about three times the scan, because each Hann evaluation of $(\mathcal A,\mathcal A',\mathcal A'')$ costs about eight times a value (83 µs against 10 µs at 562 rotations, three rectangular derivative sums each); the energy $E$ costs 0.11 ms per window. Algorithm 6 tracked 40 of the 48 windows without a scan, 38 of them with the certificate below the floor, at a median 3.1 ms, and ran a full pursuit in 8 (the first window, 4 interval rescans and the 3 windows at the onset, detected by (54)), at a median 31 ms. Over the whole run its mean cost equals that of the scan, with the accuracy of the pursuit. Eighteen track identities lasted 8 windows or more, for 16 partials: the second and third harmonics of the first note restarted with new identities at the onset. Eleven further identities lived for 1–4 windows while the onset transient was modelled by tones. The second harmonic of the second note (660.9 Hz) lies only 0.24 bins from the third harmonic of the first; the scoring counts a pair this close as recovered by one report. These results come from a synthetic, mostly stationary signal that suits a tone model; pursuit and tracking have not yet been measured on the recording above.

---

## 8. Discussion and limitations

**Circle rates are not the spectrum.** A natural hope is that the rates $\beta$ of the circles are themselves the frequencies of the sound, so that grouping them would yield the spectrum without any transform. Section 2.6 shows why this fails: a rate is twice an inscribed angle of the anchor triangle, divided by a duration, and the triangle's shape depends on the aperture. The measurements agree. On a synthetic test the rates clustered well below the true tone, and seeding the peak search from grouped rates recovered only 41–43% of the peaks and was slower than the scan. The frequencies of the signal are obtained from the transform of the circles, through the interference of all rotations (Remark 2), not from the rate of any single one.

**Peak selection.** The maxima of $\lvert\mathcal A\rvert$ are exact, but their heights are not the components' heights: the window kernel shares every level with the neighbours (Tables 1 and 2), and a selection of maxima trades accuracy among the peaks it keeps. A joint tone fit makes that trade explicit and measurable: Proposition 8 accounts for the unexplained energy and bounds it at every frequency, and the significance (49) decides which components the data support. Its separation of tones inside one main lobe rests on the tone model, not on the transform; gliding partials and onsets are not tones, and the fit spends components on them.

**Speed.** One exact continuous query is cheaper than the sampled-curve baselines and 3.3–3.5× cheaper than the per-interval evaluation of the same triplet curve, which calls trigonometric functions on every interval. A complete, fixed-bin spectrum of the *samples* remains much faster with an FFT. The advantage of the AFT is exactness and a free frequency axis, not asymptotic complexity.

**Model.** The AFT is the transform of the fitted curve. The fit is exact within tolerance at the samples only. Content between samples and above the source Nyquist frequency is produced by the circles, not recovered information; by Corollary 3 it is set by the slope jumps and value jumps at the joins. Rounded circles leave small value jumps at joins, which cause the $z^{-1}$ tail of Proposition 6.

**Future work.**

- Rotate and add in time: sliding windows that update only the rotations entering and leaving.
- Circle fits with continuous joins, which remove the $\omega^{-1}$ tails of Corollary 3.
- Tighter completeness certificates. The bound (48) is conservative when the residual is broadband, because it allows the whole residual energy at one frequency.
- Chirp atoms, tones with a linear frequency change, for gliding partials and onsets, with the same closed-form inner products.
- A first-derivative kernel: the Gauss–Newton step of Section 6.5 needs $\mathcal A$ and $\mathcal A'$ only, while the current evaluation also forms $\mathcal A''$.
- Pursuit and tracking measured on recordings, including the windows of Section 7.
- Circle determinations whose angular rates carry physical frequency. By (16), an aperture equal to the largest magnitude of the central difference makes the rate exact at a symmetric crest of a pure tone; whether any aperture rule achieves this away from crests is open.
- Many frequencies at once: by (25), the AFT is a sum of simple poles at $\pm\beta_k$ with exponential numerators. Whether this structure admits a fast multi-frequency evaluation with a stated error bound is open.

---

## 9. Conclusion

The geometric construction of Section 2 turns sampled data into a continuous piecewise-rotational curve. Every rotation follows from elementary geometry: its rate is twice an inscribed angle of the anchor triangle divided by a duration, its radius is fixed by the law of sines, and a line is its limit of infinite radius. The Fourier transform of this curve, the Angular Fourier Transform, has an exact closed form at every real frequency. Each arc contributes a term fixed by its centre height, its rate and its endpoints, and the time distance between arcs enters only as a phase. The Fast Angular Fourier Transform evaluates that form by rotating and adding the contributions of the fitted arcs, with no quadrature, no dense resampling and no transcendental work per rotation. Dense evaluation is therefore not a route to a more exact version of the same spectrum: it produces a sampled approximation with spectral images and join terms that are absent from the direct AFT. Combined with analytic derivatives, FAFT yields relevant frequencies on a continuous axis and, in the reported experiments, substantially reduces the cost of the continuous-model baselines. Because the window kernel shares every peak's level with its neighbours, relevant frequencies are selected jointly: a closed-form tone fit gives each component's own height, accounts for the energy left unexplained with a bound at every frequency, and can be carried from window to window without rescanning. It is not claimed to replace the FFT for computing the complete fixed-bin spectrum of the source samples.

---

## References

1. R. Cecchini, "SinWhisky e FLT," technical whitepaper (in Italian), 2014. Included in the repository as `docs/SinWhisky e FLT.pdf`.
2. J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series," *Mathematics of Computation* 19 (1965) 297–301.
3. J. O. Smith III, *Mathematics of the Discrete Fourier Transform (DFT), with Audio Applications*, 2nd ed., W3K Publishing, 2007 (online edition at CCRMA, Stanford).
4. G. Goertzel, "An algorithmic method for the evaluation of finite trigonometric series," *The American Mathematical Monthly* 65 (1958) 34–35.
5. H. S. M. Coxeter, *Introduction to Geometry*, 2nd ed., Wiley, 1969.
6. F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform," *Proceedings of the IEEE* 66 (1978) 51–83.
7. M. Unser, "Splines: a perfect fit for signal and image processing," *IEEE Signal Processing Magazine* 16 (1999) 22–38.
8. M. J. Lighthill, *An Introduction to Fourier Analysis and Generalised Functions*, Cambridge University Press, 1958.
9. R. N. Bracewell, *The Fourier Transform and Its Applications*, 3rd ed., McGraw-Hill, 2000.
10. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
11. W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery, *Numerical Recipes: The Art of Scientific Computing*, 3rd ed., Cambridge University Press, 2007.
12. D. C. Rife and R. R. Boorstyn, "Single-tone parameter estimation from discrete-time observations," *IEEE Transactions on Information Theory* 20 (1974) 591–598.
13. J. Li and P. Stoica, "Efficient mixed-spectrum estimation with applications to target feature extraction," *IEEE Transactions on Signal Processing* 44 (1996) 281–295.
14. K. Levenberg, "A method for the solution of certain non-linear problems in least squares," *Quarterly of Applied Mathematics* 2 (1944) 164–168.
15. D. W. Marquardt, "An algorithm for least-squares estimation of nonlinear parameters," *Journal of the Society for Industrial and Applied Mathematics* 11 (1963) 431–441.
16. G. H. Golub and V. Pereyra, "The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate," *SIAM Journal on Numerical Analysis* 10 (1973) 413–432.
17. J. A. Högbom, "Aperture synthesis with a non-regular distribution of interferometer baselines," *Astronomy and Astrophysics Supplement Series* 15 (1974) 417–426.
18. S. G. Mallat and Z. Zhang, "Matching pursuits with time-frequency dictionaries," *IEEE Transactions on Signal Processing* 41 (1993) 3397–3415.
19. Y. C. Pati, R. Rezaiifar and P. S. Krishnaprasad, "Orthogonal matching pursuit: recursive function approximation with applications to wavelet decomposition," *Proceedings of the 27th Asilomar Conference on Signals, Systems and Computers*, 1993, 40–44.
20. R. J. McAulay and T. F. Quatieri, "Speech analysis/synthesis based on a sinusoidal representation," *IEEE Transactions on Acoustics, Speech, and Signal Processing* 34 (1986) 744–754.
21. X. Serra and J. O. Smith III, "Spectral modeling synthesis: a sound analysis/synthesis system based on a deterministic plus stochastic decomposition," *Computer Music Journal* 14(4) (1990) 12–24.

### Reproducibility notes

The mathematical definition of the method is contained in Definitions 1–3, Propositions 1–8, Corollaries 1–3, the kernel (37)–(38), the inner products (45)–(46) and (51)–(53), and Algorithms 1–6. Sections 5.2 and 5.3 state the resonance guard and the numerical-stability thresholds used near DC and resonance, and Section 7 reports the fitting tolerance, support limit, sample format, window size, implementation language and version, and benchmark protocol. The reference implementation is `src/faft.rs` of the SinWhisky.rs crate, with the selection of Sections 6.2–6.7 in `src/faft/pursuit.rs`; the synthetic comparison of Table 3 is `examples/pursuit_bench.rs`, which prints the aggregate snapshot. Its tests check the closed forms against independent Gauss–Legendre quadrature, including a flat arc at $10^{-12}$ relative, the derivatives against quadrature of the moments of (33) and against finite differences, dense evaluation against Proposition 6 including images and join terms, and FAFT against the per-interval evaluation on the triplet curve. For Section 6 they check the kernel against the exact transform of a tone, the widths and sidelobes of Table 1, the energy against independent integration, the certificate (48) at 4,000 frequencies, sidelobes explained by their tone, a −30 dB tone hidden on a strong flank, tones 1.0 and 1.5 bins apart separated to $10^{-6}$ of a bin, shared and isolated peak ranges, and stable track identities through a glide and an onset (`tests/faft_pursuit.rs`). They also check the geometric statements of this paper: the rates as doubled inscribed angles and the crest relation (16), the endpoint form (24), the infinite-radius limit of Section 2.5, and the element and array factors of (30). Figure 1 is generated by `tools/faft_geometry_figure.py`, which asserts (8) for its example. The method can therefore be interpreted and implemented from the manuscript alone.
