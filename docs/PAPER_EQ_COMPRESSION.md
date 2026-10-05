# Equalization and Compression with the Angular Fourier Transform

**Riccardo Cecchini** — riccardo.cecchini594@edu.unito.it

*Draft, 5 October 2026. Extension of "The Angular Fourier Transform for Obtaining a Continuous Spectrum" [1].*

> **Status of this draft.** Sections 2–4 derive the methods. The identities they state are checked against independent computations by `tools/eq_compression_checks.py`, which also produces Table 1 and the figures quoted in Sections 2.4 and 4.2. A first implementation in the crate (Sections 6.1–6.2) covers the banded fit, exact analog filtering, the divider, equalization, extrusion and the layered coder SWL2 (LPC correction with an LMS cascade, frequency-gradient groups, SinWhisky anchors, automatic parameter choice), with byte counts on three 5-second excerpts. No listening result is claimed, and three excerpts of one recording do not establish a general compression result. Earlier measurements in this project found no material coding gain from local circle descriptors (Section 5). Still to be written: figures, and the remaining experiments of Section 6.

---

## Abstract

The Angular Fourier Transform (AFT) gives the exact spectrum of a continuous circle curve at every real frequency, and its joint tone fit gives every selected component its own height together with a bound on what is left [1]. This paper applies that machinery to two operations on sound: dividing a track into frequency bands, from which equalization follows, and compressing it. Dividing a track by frequency is the same zero-sum problem as selecting its peaks. Masking a spectrum assigns each tone's window leakage to whichever band it falls in: with a rectangular window, a tone two bins above an edge leaves 2.5% of its windowed energy below it, and even with a Hann window a tone one bin away leaves 4%. Assigning fitted components whole instead keeps the tonal part of each band free of the other bands' tones, makes the division complete by construction, and admits a brick-wall edge for tones without ringing. Ordering the components by frequency makes the joint fit banded, because their coupling depends only on their frequency distance. Its exact solution is then one ascending pass, which already yields the explained energy and the residual bound, followed by one descending pass, which yields the amplitudes: detecting from low to high frequencies and correcting from high to low is the factorization of the fit, not an approximation of it. For the part of the curve that no tone explains, any rational analog filter can be applied to a circle curve exactly. The filter state is carried from arc to arc by the rotate-and-add chain of FAFT with complex rates, and the output is a curve of spiral rotations whose AFT is again a closed form. An equalizer is then a divider followed by gains. For compression we propose a layered, closed-loop representation in which the tonal part of a band is a set of partial tracks described by frequencies and their gradients instead of by samples. The gradients are estimated in one window from the AFT and its first two derivatives, with every inner product in closed form, and a decoded track is one circle traversed with stepwise angular speed. The band residual is sampled at the band's own Nyquist rate, and a final layer writes only the samples whose error exceeds a tolerance $\tau$, as small integers of step $2\tau$, so that every decoded sample is within $\tau$ of the source. Low bands are where this should pay first: their Nyquist rate is low, and their partials are few and slow. No bitrate is measured in this draft.

---

## 1. Introduction

The AFT of [1] is the Fourier integral of a continuous curve built from circles fitted to the samples, available in closed form at every real angular frequency $\omega$. Section 6 of [1] showed that the maxima of that transform are not yet a list of components. Through the window kernel, every peak's height is shared with its neighbours, so choosing peaks one at a time is a zero-sum selection: each missed or spurious peak biases the others. A joint fit of all selected tones resolves the sharing, and the residual energy bounds what is left at every frequency.

Two operations on sound depend on knowing which content lies at which frequency. An *equalizer* changes the level of a track as a function of frequency. A *compressor* spends bits on the content that is there, and none on what is not. Both start from the same act: dividing a track into frequency ranges. The simplest equalizer is a *channel divider*, or crossover, which splits a track into bands that add back to the original, followed by one gain per band [2, 3]; a subband coder is a divider followed by one coder per band [4].

This paper starts from the divider. Section 2 shows that dividing a track by frequency inherits the zero-sum problem of [1]. The energy observed near a band edge cannot be assigned to a band until it is explained by components, and a component near an edge spreads its kernel across it. The joint fit of [1] divides components instead of the transform. Section 2.4 shows that the order in which components are detected matters for cost but not for the result: with components sorted by frequency, the joint fit is solved exactly by one pass from low to high frequencies and one pass back. Section 2.7 divides the remaining non-tonal curve with analog filters applied exactly to the circles. Section 3 derives equalization and band extraction ("extrusion") from the divider, and Section 4 turns the divided representation into a compression scheme. Section 5 relates the proposal to earlier work and to earlier measurements in this project, Section 6 reports a first implementation and lists the experiments that must precede any claim, and Section 7 discusses limitations.

### 1.1 Contributions and their status

| Contribution | Section | Status |
|---|---|---|
| Leakage of spectral masking across a band edge (Table 1) | 2.2 | Exact property of the kernels; computed |
| Component divider: completeness, selectivity, residual bound | 2.3 | Derived; implemented and tested (Section 6.1) |
| Two-way (ascending, descending) banded solution of the joint fit | 2.4 | Derived; implemented; truncation computed for one synthetic series |
| Tonal tracks are circle curves whose rates are physical frequencies | 2.6 | Derived |
| Exact analog filtering of circle curves; spiral rotations | 2.7 | Derived; implemented; checked against Runge–Kutta integration and quadrature |
| Equalization and extrusion from the divider | 3 | Derived; implemented |
| Gradient tones in closed form from AFT derivatives | 4.2 | Derived; checked against direct least squares |
| Gradient tracks rendered on one circle | 4.3 | Derived; implemented as quadratic-phase (constant-gradient) segments in gradient groups (Section 6.2), not Hermite gradients |
| Layered closed-loop coding with integer corrections | 4.5 | Implemented as SWL2 with LPC, an LMS cascade and automatic choices; error contract tested; bytes measured on three excerpts (Tables 3–4) |

### 1.2 Notation

This paper uses the notation of [1]. Equation, section, definition, proposition, corollary, table and algorithm numbers prefixed with F refer to [1]: (F44) is its equation (44), Proposition F8 its Proposition 8. New symbols are listed below.

| Symbol | Meaning | Unit |
|---|---|---|
| $B$, $b$ | Number of bands; band index | — |
| $I_b$ | Band $b$, the interval $[\omega_{b-1},\omega_b)$ of nonnegative frequencies | rad/sample |
| $y_b$, $\mathcal T_b$, $r_b$ | Track of band $b$; its tonal part; its residual part | amplitude |
| $\psi_b(\omega)$ | Crossover weight of band $b$ | — |
| $\Delta$ | Bin width $2\pi/T$ | rad/sample |
| $\Lambda(s)$ | Fraction of a tone's windowed energy beyond an edge $s$ bins away | — |
| $P$, $p$ | Number of fitted components in a window; component index, in ascending frequency | — |
| $G$, $L$, $z$ | Gram matrix of the joint fit (F45); its Cholesky factor; coefficients of the ascending pass | —, —, amplitude |
| $\kappa$, $m$ | Reach of the banded fit, in bins; resulting bandwidth, in components | —, — |
| $h$, $H$ | Impulse response and frequency response of an analog filter | 1/sample, — |
| $p_j$, $\eta_j$, $\xi_j$ | Pole, residue and state of an analog filter | 1/sample, 1/sample, amplitude·samples |
| $\lambda=\sigma+i\beta$ | Complex rate of a spiral rotation | 1/sample |
| $\Gamma(\omega)$, $\Gamma_b$ | Equalizer response; band gain | — |
| $\varrho$, $\gamma$ | Relative amplitude gradient and frequency gradient of a partial | 1/sample, rad/sample² |
| $\tau$ | Tolerance of the final coding layer | amplitude |
| $f$ | Fraction of samples written by the final layer | — |

---

## 2. Dividing a track by frequency

### 2.1 Dividers

**Definition 1 (band divider).** Let $0=\omega_0 < \omega_1 < \cdots < \omega_B=\infty$ be band edges and $I_b=[\omega_{b-1},\omega_b)$, $b=1,\ldots,B$, the bands on the nonnegative frequency axis; a real track has a conjugate-symmetric transform, so negative frequencies follow. A band divider maps a curve $y$ to tracks $y_1,\ldots,y_B$ such that

$$
\sum_{b=1}^By_b(t)=y(t)\qquad\text{for every }t, \tag{1}
$$

and is judged by how much of each $y_b$ lies outside $I_b$.

We call (1) *completeness* and the second requirement *selectivity*. Completeness is easy to guarantee, by defining the last track as $y-\sum_{b < B}y_b$, so a divider is judged by its selectivity. Completeness is nevertheless what makes a divider useful for equalization: with unit gains, the equalized track is the original exactly (Section 3). The edges are real numbers, not bins. The AFT is defined above the source Nyquist frequency $\pi$, where the content of the curve is produced by the circles rather than recorded (Section F8); it belongs to the last band.

A linear time-invariant divider is a set of filters with frequency responses $H_b$ and $\sum_bH_b=1$; a tone at $\omega$ reaches track $b$ with gain $H_b(\omega)$. Its selectivity is limited by the uncertainty principle: a filter whose transition is narrower than $\delta\omega$ has an impulse response longer than about $2\pi/\delta\omega$, and rings for that long [5]. The dividers compared below are not time invariant. They act within analysis windows.

### 2.2 Masking a spectrum leaks across the edge

The usual block divider computes a windowed transform, multiplies it by the indicator of each band, and resynthesizes by overlap-add [6]. On the continuous axis, by (F43), a tone of frequency $\omega_q$ contributes $cK(\omega-\omega_q)$ to the centred transform, plus its image at $-\omega_q$. If an edge lies $s$ bins below $\omega_q$, the part of the kernel beyond the edge is assigned to the lower band. Neglecting the image, the fraction of the tone's windowed energy that crosses the edge is

$$
\Lambda(s)=\frac{\int_{s\Delta}^{\infty}K(\nu)^2d\nu}{\int_{-\infty}^{\infty}K(\nu)^2d\nu}. \tag{2}
$$

**Proposition 1 (leakage of a masked edge).** For the rectangular weight, $\Lambda(s)=1/(2\pi^2s)+O(s^{-2})$. For the Hann weight, $\Lambda(s)=O(s^{-5})$.

*Proof.* In bins $x=\nu/\Delta$, the kernels (F42), divided by $K(0)$, are $k_{\mathrm{rect}}(x)=\operatorname{sinc}(\pi x)$ and $k_{\mathrm{Hann}}(x)=\operatorname{sinc}(\pi x)+\tfrac12\operatorname{sinc}\bigl(\pi(x-1)\bigr)+\tfrac12\operatorname{sinc}\bigl(\pi(x+1)\bigr)=-\sin(\pi x)/\bigl(\pi x(x^2-1)\bigr)$. By Parseval's theorem their squares integrate to $1$ and $3/2$. The square of the rectangular kernel has the mean $1/(2\pi^2x^2)$ over each period, which integrates to $1/(2\pi^2s)$ beyond $s$, with an oscillating remainder of order $s^{-2}$. The Hann kernel decays like $\lvert x\rvert^{-3}$, so its square integrates to $O(s^{-5})$. $\square$

| Edge distance $s$ (bins) | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| Rectangular | 11.3% (−9.5 dB) | 4.86% (−13.1 dB) | 2.50% (−16.0 dB) | 1.26% (−19.0 dB) | 0.63% (−22.0 dB) | 0.32% (−25.0 dB) |
| Hann | 20.0% (−7.0 dB) | 4.10% (−13.9 dB) | $2.6\cdot10^{-4}$ (−35.9 dB) | $6.9\cdot10^{-6}$ (−51.6 dB) | $2.1\cdot10^{-7}$ (−66.8 dB) | $6.5\cdot10^{-9}$ (−81.9 dB) |

*Table 1. Fraction $\Lambda(s)$ of a tone's windowed energy that lies beyond a band edge $s$ bins away, from (2) and the kernels (F42), computed by Gauss–Legendre quadrature (`tools/eq_compression_checks.py`).*

Masking divides the transform, not the content. A tone one bin from an edge sends about 4–5% of its windowed energy into the neighbouring band with either window. The rectangular window still leaks at −16 dB from two bins away and decays only like $1/s$; the Hann window confines the leakage to about two bins, as it confines the coupling of peaks (Table F2). Each band then holds a fragment of the tone, whose size changes with the tone's distance from the edge. A divider meant to isolate a range of frequencies in its own track, with the rest in another, must assign each tone to one band in full.

### 2.3 Dividing components instead of the transform

Let a window be fitted as in Sections F6.4–F6.6: $y=\sum_{p=1}^Px_p+r$ on the window, where $x_p=\alpha_pC_p+\beta_pS_p$ are the fitted tones and $r$ is the residual curve. Crossover weights are functions $\psi_b(\omega)\ge0$ with

$$
\sum_{b=1}^B\psi_b(\omega)=1\qquad\text{for every }\omega\ge0. \tag{3}
$$

The brick-wall choice is the indicator of $I_b$. A smooth choice, for example a raised cosine, moves from 0 to 1 over a transition around each edge.

**Definition 2 (component divider).** Given a residual divider $r=\sum_br_b$ (Section 2.7), the component divider assigns

$$
y_b=\sum_{p=1}^P\psi_b(\omega_p)x_p+r_b. \tag{4}
$$

**Proposition 2 (component divider).** (i) The component divider is complete (1) for any fit, any weights satisfying (3) and any complete residual divider. (ii) With brick-wall weights, the tonal part of $y_b$ contains no fitted component outside $I_b$, and its transform outside $I_b$ consists only of the kernels (F43) of its own tones. (iii) Any component that the fit does not represent is in the residual, whose normalized transform is at most $\sqrt{E_r/W}$ at every real frequency.

*Proof.* (i) Sum (4) over $b$ and use (3). (ii) By construction and (F43). (iii) This is the bound (F48) of Proposition F8, the certificate of [1]. $\square$

Three consequences follow. First, the energy near an edge is assigned by its source. The joint fit has already removed from every height the leakage of the neighbours (Section F6.4), and a fitted tone goes to its band with its whole kernel, sidelobes included. The tails of a band's own tones that reach beyond its edges belong to those tones, not to another band's content, and they vanish from the track when consecutive windows are overlapped (Section 2.6).

Second, the selectivity of the tonal part does not depend on the width of the transition. A brick-wall edge, which as a linear filter would ring indefinitely, is admissible here: a gain multiplies an entire atom of the window, and nothing is truncated in frequency.

Third, the division is as good as the fit. Two fitted components that represent one tone on both sides of an edge would split it; significance pruning (F49) removes such duplicates. Components that the fit does not explain are left to the residual divider, and Proposition 2 (iii) bounds their level. A tone that glides across an edge changes track in the window where it crosses; smooth weights spread the change over the windows during which it moves through the transition. Completeness holds in either case.

### 2.4 The two-way sweep

The joint fit (F45)–(F46) couples every pair of components through the kernel, so the precision with which each frequency is detected depends on the detection of all the others. A natural procedure is therefore to detect components from low to high frequencies and then correct them from high to low. This section shows that, for the amplitudes, this order is exactly the structure of the joint solution.

Sort the components by frequency, $\omega_1 < \cdots < \omega_P$. The cosine system of (F45) has the Gram entries $G_{pq}=\bigl(k(\omega_p-\omega_q)+k(\omega_p+\omega_q)\bigr)/2$, and the sine system the difference of the same two terms. Both depend on frequency differences through the normalized kernel $k$, which decays: the Hann kernel has its first null at 2 bins, its sidelobes are below −31 dB beyond it, and they decay like the cube of the distance (Table F1). In frequency order, $G$ is therefore numerically banded. Keeping only the entries of components at most $\kappa$ bins apart gives a matrix $G^{(\kappa)}$ with a variable band: row $p$ holds entries from the lowest component within $\kappa$ bins of $\omega_p$ up to the diagonal. Its bandwidth $m$ is the largest number of components within $\kappa$ bins above any component.

**Proposition 3 (two passes).** Let $G=LL^{\mathsf T}$ be the Cholesky factorization of either system of (F45), with the components in ascending frequency, and $b$ its right side (F46).

(i) The ascending pass $z=L^{-1}b$ (forward substitution, $p=1,\ldots,P$) gives $z_p=\langle y,o_p\rangle$, where $o_p$ is the normalized part of atom $p$ orthogonal to all atoms of lower frequency. The energy explained by the components up to $\omega_p$ is $W\sum_{q\le p}z_q^2$, and the residual energy (F47) is

$$
E_r=E-W\bigl(\lVert z_c\rVert^2+\lVert z_s\rVert^2\bigr), \tag{5}
$$

with $z_c$ and $z_s$ the coefficients of the cosine and sine systems. The bound (F48) is therefore known at the end of the ascending pass, before any amplitude.

(ii) The descending pass $\alpha=L^{-\mathsf T}z$ (back substitution, $p=P,\ldots,1$) gives the jointly fitted amplitudes. The highest component is final at the turn, $\alpha_P=z_P/L_{PP}$, and each later step completes the next lower one.

(iii) If $G$ has bandwidth $m$, so has $L$. The factorization costs $O(Pm^2)$, each pass $O(Pm)$, and the diagonal of $G^{-1}$, which the significance (F49) needs, follows from one more descending recursion within the band [8] in $O(Pm^2)$.

*Proof.* Let $V$ be the quasi-matrix whose columns are the atoms, so that $G=V^{\mathsf T}V$ and $b=V^{\mathsf T}y$ in the inner product of (F45). The columns of $Q=VL^{-\mathsf T}$ are orthonormal. Since $L^{-\mathsf T}$ is upper triangular, column $p$ of $Q$ is a combination of atoms $1,\ldots,p$, and since $Q^{\mathsf T}V=L^{\mathsf T}$ is upper triangular, it is orthogonal to atoms $1,\ldots,p-1$. Hence $Q$ is the Gram–Schmidt basis in ascending order, and $z=L^{-1}V^{\mathsf T}y=Q^{\mathsf T}y$. By (F47), $E_r=E-Wb^{\mathsf T}G^{-1}b=E-W\lVert L^{-1}b\rVert^2$ for each system. (ii) is back substitution. For (iii), the Cholesky factor of a band matrix has the same band [7], and the entries of $G^{-1}$ within the band satisfy a recursion in $L$ that runs from the last row to the first [8]. $\square$

The solution does not depend on the order in which components are eliminated: any order needs one pass in each direction. Frequency order is the order in which both passes are *local*. Because the coupling of two components is a function of their frequency distance, it is the ordering that keeps the factor banded, as a bandwidth-reducing reordering does for a sparse matrix [9]. The two passes also read as detection. The ascending pass explains the curve from the lowest component up: $z_p$ is what component $p$ adds to the components below it, the running sum of $Wz_p^2$ is the energy explained below each frequency, and the pass ends with the residual bound. The descending pass removes from each component the leakage of the components above it, which the ascending pass could not yet see. Detection from low to high and correction from high to low is therefore not an approximation of the joint fit: it is the joint fit.

The fit of [1] solves components farther apart than 4 bins (Hann) as separate clusters, with dense algebra inside each cluster (Section F6.5). Low notes form long chains. At 44.1 kHz and $N=1024$, the harmonics of a 55 Hz note are 1.28 bins apart, so twenty of them form one cluster of twenty, while each harmonic has at most three others within 4 bins above it. The banded solve treats a chain of any length at a cost linear in its length. The reach must be chosen against an accuracy, because the leakage of many weak neighbours accumulates. For twenty harmonics of 55 Hz with cosine amplitudes $\cos\phi_k/k$ and random phases $\phi_k$, truncating the Gram matrix beyond 4, 8 and 16 bins changed the fitted amplitudes by $4.7\cdot10^{-3}$, $8.4\cdot10^{-4}$ and $8.3\cdot10^{-5}$ in relative root-mean-square value, compared with the full solve (`tools/eq_compression_checks.py`).

*Truncation.* If $G^{(\kappa)}$ replaces $G$ and remains positive definite, the amplitudes $\tilde\alpha$ satisfy $G^{(\kappa)}(\tilde\alpha-\alpha)=\bigl(G-G^{(\kappa)}\bigr)\alpha$, so $\lVert\tilde\alpha-\alpha\rVert\le\lVert(G^{(\kappa)})^{-1}\rVert\lVert G-G^{(\kappa)}\rVert\lVert\alpha\rVert$. Identity (5) holds only for the exact solution. The bound (F48) holds for any amplitudes, provided $E_r$ is evaluated from the full quadratic form $E-2W\tilde\alpha^{\mathsf T}b+W\tilde\alpha^{\mathsf T}G\tilde\alpha$ of each system.

*Frequencies.* The Gauss–Newton system of Section F6.5 adds one derivative atom per component, with inner products (F51)–(F52) that are again functions of frequency differences, through $k'$ and $k''$. In frequency order its matrix is block banded with blocks of three, Levenberg–Marquardt damping adds to its diagonal only, and each step is one ascending and one descending block pass.

```text
Algorithm 1  TWO-WAY(prepared window, ω_1 < … < ω_P, reach κ)
  for each of the cosine and sine systems of (F45):
      for p = 1 … P:                                  ▷ ascending: factor and detect
          b_p ← Re a(ω_p)  or  −Im a(ω_p)                 ▷ (F46): one FAFT value
          G_pq ← (F45) for q ≤ p with ω_p − ω_q ≤ κ bins  ▷ kernel only
          row p of L ← band Cholesky
          z_p ← (b_p − Σ_q L_pq z_q) / L_pp
          explained ← explained + W·z_p²                  ▷ energy up to ω_p
  E_r ← E − explained;  bound ← √(E_r/W)              ▷ (5), (F48): no amplitude yet
  for each system, for p = P … 1:                     ▷ descending: correct
      α_p ← (z_p − Σ_{q>p} L_qp α_q) / L_pp
      (G⁻¹)_pp ← selected inversion within the band       ▷ significance (F49)
  return amplitudes, significances, E_r, bound
```

### 2.5 Bands with their own windows

The resolution $2\pi/T$ is the same number of hertz at every frequency, so a window long enough to resolve the harmonics of a bass note is far longer than one that follows the onsets of the upper bands. A divider can give each band its own window duration $T_b$. With different windows a single joint fit no longer exists, and the bands are fitted in turn: band 1 on $y$, band 2 on $y$ minus the synthesized track of band 1 (Section 2.6), and so on upwards, then back downwards, each band refitted on $y$ minus the current tracks of all the others. This is backfitting [10]. With the frequencies fixed and one common inner product, the exact joint solution is the block form of Proposition 3, and an ascending-then-descending backfitting sweep is its iterative counterpart, symmetric block Gauss–Seidel, which converges for every positive definite Gram matrix [11]. If each band is wider than the kernel's reach, only the components near each edge couple adjacent bands. With windows that differ by band, the band fits are projections in different inner products, and convergence is not established here (Section 6).

### 2.6 From windows to tracks

Within one window, (4) divides the curve. Over a whole track, windows of duration $T$ advance by a hop of $T/2$, and the fitted tones of window $k$ are weighted by its Hann window $w_k$. The periodic Hann window satisfies

$$
\sum_kw_k(t)=1 \tag{6}
$$

wherever two windows overlap, because consecutive windows are $\tfrac12-\tfrac12\cos$ and $\tfrac12+\tfrac12\cos$ of the same angle. The tonal part of band $b$ is

$$
\mathcal T_b(t)=\sum_kw_k(t)\sum_p\psi_b(\omega_{k,p})x_{k,p}(t), \tag{7}
$$

where $x_{k,p}$ is tone $p$ of window $k$, and the residual of the track is $r=y-\sum_b\mathcal T_b$. Completeness (1) holds over the whole track because it holds for every term. A stationary tone fitted with the same amplitude and phase in consecutive windows is reconstructed exactly by (6), so the kernel tails of Proposition 2 (ii), which belong to the analysis, are absent from the track.

**Proposition 4 (a tonal track is a circle curve).** On each half-hop interval, $\mathcal T_b$ is a sum of stacked rotations (Definition F1) with zero centre height and zero slope, whose rates are the fitted frequencies shifted by $0$ and $\pm\Delta$. Hence $\mathcal T_b$ is a circle curve (Definition F2) whose rates are physical frequencies of the fit, FAFT evaluates its transform exactly, and $r$ is a circle curve on the common refinement of the supports of $y$ and of the half-hop grid.

*Proof.* With $u$ the time from the window centre, the Hann weight is $w=\tfrac12+\tfrac12\cos\Delta u$, so

$$
w(u)\cos(\omega u+\phi)=\tfrac12\cos(\omega u+\phi)+\tfrac14\cos\bigl((\omega+\Delta)u+\phi\bigr)+\tfrac14\cos\bigl((\omega-\Delta)u+\phi\bigr), \tag{8}
$$

three rotations on the window's support. Cropping them at the window centre (Section F3.5) gives rotations on the two half-hop intervals of the window, and on each such interval the rotations of the two overlapping windows are stacked. Two circle curves are subtracted on the common refinement of their supports, obtained by cropping each rotation at the other's boundaries, which again gives a circle curve. $\square$

Section F2.6 showed that the rates of determined circles are geometric, not physical frequencies. The rates in (8) are physical frequencies, because these rotations are synthesized from the fit rather than determined from three samples. Both kinds enter the same transform.

### 2.7 Dividing the residual exactly: analog filters on circle curves

The residual holds no component above the level of its bound, but it carries noise, onsets and the joins of the circle curve (Corollary F3). It is divided by complementary filters, $r_b=h_b\ast r$ for $b < B$ and $r_B=r-\sum_{b < B}r_b$, so that (1) holds whatever the filters are [3]. Linear filters have transition bands and ring, as in Section 2.1; the residual is the part of the track where this matters least, because it holds no strong tones. What a circle curve adds is that the filtering can be exact.

Let a filter have distinct poles $p_j$ with $\operatorname{Re}p_j < 0$ and residues $\eta_j$, with impulse and frequency responses

$$
h(t)=\eta_0\delta(t)+\sum_j\eta_je^{p_jt}\quad(t\ge0),\qquad H(\omega)=\eta_0+\sum_j\frac{\eta_j}{i\omega-p_j}. \tag{9}
$$

Its output is $h\ast y=\eta_0y+\sum_j\eta_j\xi_j$, where each state solves $\xi_j'=p_j\xi_j+y$. Butterworth and Linkwitz–Riley crossovers [2] and the analog prototypes of parametric equalizer sections [12] have this form; a real filter has its poles in conjugate pairs.

**Proposition 5 (filtering a rotation).** Let a rotation of a circle curve $y$ start at $s$, with duration $L$ and local transform $\Phi$ (F21). For $0\le u\le L$,

$$
\xi_j(s+u)=e^{p_ju}\bigl[\xi_j(s)+\Phi_u(-ip_j)\bigr], \tag{10}
$$

where $\Phi_u$ is the transform (F21) of the rotation cropped to $[s,s+u)$, evaluated at the complex frequency $-ip_j$. In particular, the states pass from support to support as

$$
\xi_j(s_{k+1})=e^{p_jL_k}\bigl[\xi_j(s_k)+\Phi_k(-ip_j)\bigr]. \tag{11}
$$

*Proof.* Variation of constants gives $\xi_j(s+u)=e^{p_ju}\xi_j(s)+\int_0^ue^{p_j(u-v)}y(s+v)dv$ [13]. The integral equals $e^{p_ju}\int_0^uy(s+v)e^{-p_jv}dv$, and $e^{-p_jv}=e^{-i\omega v}$ at $\omega=-ip_j$. The moments $E_\ell$ of (F18) are entire functions of their offset, so (F21) holds at complex frequencies. $\square$

Recursion (11) is the rotate-and-add chain of FAFT (F29), run in time and written in Horner form: the unit phasor $q^{L_k}$ becomes the decaying rotation $e^{p_jL_k}$. It costs $O(R)$ per pole for $R$ rotations, with no sampling, no step size and no quadrature, and inside a support (10) gives the output at any position, which is how a dense rendering (zoom) of the filtered curve is obtained. The complex frequency $-ip_j$ has the nonzero imaginary part $-\operatorname{Re}p_j$, so it never equals a real rate $\pm\beta$: the endpoint form (F24) applies without a resonance guard and keeps its accuracy on flat arcs (Section F5.3). For strongly damped poles, $\lvert\operatorname{Re}p_j\rvert L\gg1$, the product in (10) should be formed with the time origin at the end of the support, so that no factor grows.

The output is no longer a circle curve. By (10), on each support it is the sum of the input's rotations scaled by the filter's response at their own rates, constant and linear terms, and terms $e^{p_ju}$ that decay at the filter's poles. Geometrically, a rotation whose radius shrinks at a constant rate is a spiral.

**Definition 3 (spiral rotation).** A spiral rotation is a tuple $(s,L,c,r,\varphi,\lambda,d)$ as in Definition F1, with a complex rate $\lambda=\sigma+i\beta$ in place of the rate:

$$
y(s+u)=c+\operatorname{Im}\bigl(\zeta_0e^{\lambda u}\bigr)+du=c+re^{\sigma u}\sin(\varphi+\beta u)+du,\qquad \zeta_0=re^{i\varphi}. \tag{12}
$$

The moving point $C+\zeta_0e^{\lambda u}$ runs along a logarithmic spiral at the constant angular speed $\beta$; $\sigma=0$ is a rotation.

**Proposition 6 (spiral curves).** (i) A filter (9) maps a curve of spiral rotations to a curve of spiral rotations with stacked supports. (ii) With $M(\mu)=\int_0^Le^{\mu u}du=(e^{\mu L}-1)/\mu$, the transform of one spiral rotation, with phase origin at its start, is

$$
\Phi(\omega)=cE_0(-\omega)+\frac{\zeta_0M(\lambda-i\omega)-\overline{\zeta_0}M(\overline\lambda-i\omega)}{2i}+dE_1(-\omega). \tag{13}
$$

(iii) For $d=0$, let $g=y-c$ and $[f]_\omega=f(L^-)e^{-i\omega L}-f(0)$ in local time. Then

$$
\Phi(\omega)=cE_0(-\omega)-\frac{[g']_\omega+(i\omega-2\sigma)[g]_\omega}{(i\omega-\lambda)(i\omega-\overline\lambda)}. \tag{14}
$$

For $\sigma=0$, (14) is the endpoint form (F24). For $\sigma\ne0$ its denominator has no real zero: a spiral has no resonance on the real frequency axis.

*Proof.* (i) On a support, the input is a sum of terms $u^ne^{\mu u}$ with $n\le1$. By (10), the state of pole $p_j$ adds $e^{p_ju}$ times a constant to $\int_0^ue^{p_j(u-v)}v^ne^{\mu v}dv$, which for $\mu\ne p_j$ is a combination of $u^{n'}e^{\mu u}$ with $n'\le n$ and of $e^{p_ju}$. Real filters and real inputs give conjugate pairs, that is, spiral rotations. (ii) Write $\operatorname{Im}w=(w-\overline w)/(2i)$ and integrate (12) against $e^{-i\omega u}$ term by term. (iii) $g$ satisfies $g''-2\sigma g'+\lvert\lambda\rvert^2g=0$. Integrating by parts twice, as in Proposition F2, gives for its transform $\Phi_g=\Phi-cE_0(-\omega)$ the equation $\bigl(\lvert\lambda\rvert^2-\omega^2-2i\sigma\omega\bigr)\Phi_g=-[g']_\omega-(i\omega-2\sigma)[g]_\omega$, and $\lvert\lambda\rvert^2-\omega^2-2i\sigma\omega=(i\omega-\lambda)(i\omega-\overline\lambda)$. $\square$

Repeated poles, as in Linkwitz–Riley crossovers, which are squared Butterworth filters, add terms $u^ne^{p_ju}$. Their transforms are the higher moments $E_n$ of (F19) at complex offsets, so the class extends to spirals with polynomial weights and keeps its closed forms.

*Complementary pairs and zero phase.* A pair $h$ and $\delta-h$ is complete by construction [3]. On a circle curve, time reversal is exact as well: reversing a rotation negates its rate. Filtering forward, reversing, filtering again and reversing back therefore gives a zero-phase filter with response $\lvert H(\omega)\rvert^2$ [14], a second two-way pass, in time instead of in frequency. On a finite track the states at both ends must be initialized, as for any forward–backward filter.

```text
Algorithm 2  DIVIDE(circle curve y, edges, weights ψ_b, residual filters h_b)
  for each window k (duration T, hop T/2):
      prepare FAFT (Algorithm F1); select components (Algorithm F5 or F6),
      solving each fit and Gauss–Newton step with TWO-WAY (Algorithm 1)
  for each band b:
      tonal_b ← Σ_k w_k Σ_p ψ_b(ω_kp) x_kp            ▷ (7); rotations, Proposition 4
  r ← y − Σ_b tonal_b                                ▷ common refinement of supports
  for b < B:  r_b ← h_b ∗ r                           ▷ (11): states chained over supports
  r_B ← r − Σ_{b<B} r_b
  return y_b = tonal_b + r_b                          ▷ Σ_b y_b = y, (1)
```

---

## 3. Equalization

### 3.1 An equalizer is a divider followed by gains

With band gains $\Gamma_b$, the graphic equalizer is $y_{\mathrm{EQ}}=\sum_b\Gamma_by_b$. With unit gains it returns $y$ exactly, by (1). This transparency is what complementary dividers guarantee and what a bank of independent peaking filters does not, because their responses overlap and their phases interact. A parametric equalizer with a continuous response $\Gamma(\omega)$, with $\Gamma(-\omega)=\overline{\Gamma(\omega)}$, acts on each fitted component as

$$
c_p\mapsto\Gamma(\omega_p)c_p, \tag{15}
$$

which multiplies the amplitude of tone $p$ by $\lvert\Gamma(\omega_p)\rvert$ and advances its phase by $\arg\Gamma(\omega_p)$. It acts on the residual through a rational filter $H$ that approximates $\Gamma$, applied exactly by Proposition 5.

(15) is the steady-state response of every linear time-invariant filter with frequency response $\Gamma$ to a stationary tone, without the filter's transient. Two consequences follow. A real $\Gamma$ changes no phase, at the latency of one window. And the sharpness of $\Gamma$ is not limited by ringing: as far as the fit separates two tones, a notch narrower than a bin removes one and keeps the other, without the long impulse response that such a notch has as a filter. For content that is not stationary, the gains of consecutive windows are interpolated by the overlap (6), so the time resolution of the equalizer is the hop.

The effect of (15) is known in closed form. By (F43), the equalized tonal part of a window has the centred transform $W\sum_p\bigl[\Gamma(\omega_p)c_pk(\omega-\omega_p)+\overline{\Gamma(\omega_p)c_p}k(\omega+\omega_p)\bigr]$, so the depth of a notch or the level of a boost can be verified at any real frequency before anything is rendered.

### 3.2 Extrusion

To *extrude* a band $I$ is to divide $y$ into the band's own track $y_I$ and the remainder $y-y_I$: a divider in which the bands outside $I$ are merged. It isolates a range of frequencies in a track of its own, for separate processing, rendering or coding (Section 4), and the remainder is exactly what is left.

**Proposition 7 (certified extrusion).** In every window, the remainder contains no fitted component inside $I$. Any component inside $I$ that is not in $y_I$ is in the residual, and its normalized level is at most $\sqrt{E_r/W}$ at every real frequency.

*Proof.* Proposition 2 (ii) and (iii), with brick-wall weights. $\square$

An extrusion is therefore certified to the level of the residual bound before the residual divider acts; how much of the in-band residual the remainder keeps is then set by the stopband of the residual filter. Because the tonal part of an extruded band is a list of frequencies and amplitudes, equalization and extrusion can also be applied to a coded stream without decoding it [15]. This is where the two halves of the paper meet.

---

## 4. Compression

### 4.1 Samples against parameters

PCM16 spends $16F_s$ bits per second per channel at the sampling rate $F_s$, whatever the content. A band with upper edge $\omega_e$, once isolated, needs a sample only every $\pi/\omega_e$ source intervals, so its sample rate drops by the factor $\pi/\omega_e$: 31.5 for a band below 700 Hz at 44.1 kHz [4, 5]. A partial is described by a few parameters per breakpoint, and that cost depends neither on the band edge nor on the sampling rate. The scheme of this section combines the three in layers: a tonal layer of partial tracks, a residual layer per band at the band's own rate, and a final correction at the source rate. Each layer is coded against the decoded output of the layers before it.

### 4.2 Frequency gradients in one window

The tones of [1] are stationary within a window. The partials of an instrument change: their frequencies glide and their amplitudes rise and decay. Over a window centred at $t_c$, with $u=t-t_c$, model a partial as a *gradient tone*

$$
x(u)=A(1+\varrho u)\cos\bigl(\omega_0u+\tfrac12\gamma u^2+\phi\bigr), \tag{16}
$$

with relative amplitude gradient $\varrho$ and frequency gradient $\gamma$, so that its instantaneous frequency is $\omega_0+\gamma u$. Expanding (16) around an analysis frequency $\omega$, with $\delta=\omega_0-\omega$, gives $x=\sum_\ell\bigl[c_\ell u^\ell e^{i\omega u}+\overline{c_\ell}u^\ell e^{-i\omega u}\bigr]$ with $c_0=Ae^{i\phi}/2$ and

$$
\frac{c_1}{c_0}=\varrho+i\delta+\cdots,\qquad \frac{c_2}{c_0}=i\Bigl(\frac\gamma2+\delta\varrho\Bigr)-\frac{\delta^2}2+\cdots, \tag{17}
$$

where the dots stand for terms of higher order in $\delta T$, $\varrho T$ and $\gamma T^2$. We fit the atoms $u^\ell e^{\pm i\omega u}$, $\ell=0,1,2$, by weighted least squares, which keeps the model linear in its coefficients. This is a local polynomial-phase model [16]. The atom $ue^{i\omega u}$ is, up to a factor $i$, the frequency derivative of $e^{i\omega u}$ that Section F6.5 uses for Gauss–Newton; the gradient fit keeps its coefficient free and adds $u^2e^{i\omega u}$.

**Proposition 8 (gradient tones in closed form).** With the complex inner product $\langle f,g\rangle=W^{-1}\int f\overline gw dt$, the normalized kernel $k=K/W$ and the normalized centred transform $a=\mathcal A^c/W$,

$$
\bigl\langle u^\ell e^{i\omega_pu},u^{\ell'}e^{i\omega_qu}\bigr\rangle=i^{\ell+\ell'}k^{(\ell+\ell')}(\omega_q-\omega_p), \tag{18}
$$

$$
\bigl\langle y,u^\ell e^{i\omega u}\bigr\rangle=i^\ell a^{(\ell)}(\omega), \tag{19}
$$

where superscripts in parentheses denote derivatives with respect to frequency. For one component the normal equations are, for $\ell'=0,1,2$,

$$
\sum_{\ell=0}^2i^{\ell+\ell'}\Bigl[c_\ell k^{(\ell+\ell')}(0)+\overline{c_\ell}k^{(\ell+\ell')}(2\omega)\Bigr]=i^{\ell'}a^{(\ell')}(\omega). \tag{20}
$$

The Gram entries need the kernel and its first four derivatives, which are elementary functions for both weights, and the data side needs $\mathcal A$, $\mathcal A'$ and $\mathcal A''$ at one frequency, which Proposition F5 gives in closed form.

*Proof.* By definition $k^{(j)}(\nu)=W^{-1}\int w(-iu)^je^{-i\nu u}dt$, so $W^{-1}\int wu^je^{-i\nu u}dt=i^jk^{(j)}(\nu)$. The integrand of (18) is $wu^{\ell+\ell'}e^{-i(\omega_q-\omega_p)u}$. Likewise $a^{(\ell)}(\omega)=W^{-1}\int yw(-iu)^\ell e^{-i\omega u}dt$ gives (19). For the real model $x=\sum_\ell\bigl(c_\ell e_\ell+\overline{c_\ell}\overline{e_\ell}\bigr)$ with $e_\ell=u^\ell e^{i\omega u}$, the least-squares conditions are $\langle y-x,e_{\ell'}\rangle=0$. The conjugate atom $\overline{e_\ell}$ has frequency $-\omega$, so by (18) its inner product with $e_{\ell'}$ is taken at $\omega_q-\omega_p=2\omega$. $\square$

From the fitted coefficients, (17) gives the first-order estimates $\hat\varrho=\operatorname{Re}(c_1/c_0)$, $\hat\delta=\operatorname{Im}(c_1/c_0)$ and $\hat\gamma=2\operatorname{Im}(c_2/c_0)-2\hat\delta\hat\varrho$. The frequency is refined first, as in Section F6.5, which keeps $\hat\delta$ small. For a gradient tone of constant amplitude analysed at its centre frequency, the error of $\hat\gamma$ is of third order in $\gamma T^2$, because the second-order terms of the expansion are real relative to $c_0$. On a 256-sample Hann window with $\gamma T^2/8=0.066$ rad, $2\operatorname{Im}(c_2/c_0)$ is within $2.6\cdot10^{-4}$ of $\gamma$ in relative terms, and the normal equations (20) agree with a direct numerical least-squares fit within $10^{-10}$, image terms included (`tools/eq_compression_checks.py`). Gradients large enough that $\gamma T^2/8$ approaches a radian are outside the expansion. They call for shorter windows, or for a nonlinear chirp atom, whose inner products involve Fresnel integrals instead of the elementary kernel. With several components, three atoms each, the entries (18) again depend on frequency differences: in frequency order the Gram matrix is block banded, and Proposition 3 applies with blocks.

### 4.3 Tracks of gradients

Across windows, the tracker of Section F6.7 links the components of consecutive windows into partials. A partial is stored as a list of *breakpoints* at window centres $t_j$, each holding a frequency, a frequency gradient, an amplitude, an amplitude gradient and a phase, $(\omega_j,\gamma_j,A_j,\varrho_j,\phi_j)$. Between breakpoints, frequency and amplitude follow the cubic Hermite interpolants of their values and gradients, and the phase is the integral of the frequency, corrected to meet the phase of the next breakpoint. If the integrated phase misses $\phi_{j+1}$ by $\Delta\phi_j$, unwrapped to $\lvert\Delta\phi_j\rvert\le\pi$ as in [17], adding $\Delta\phi_j(3v^2-2v^3)$, with $v=(t-t_j)/(t_{j+1}-t_j)$, matches the phase at both ends without changing the frequency there.

The list of breakpoints is a series of frequency gradients. Where a partial changes at a steady rate its breakpoints are redundant, and a breakpoint is removed when the interpolation from its neighbours predicts its frequency, amplitude and phase within their quantization steps. This is the polyline simplification of [18] applied to a track, and the counterpart of the checked merging of adjacent circles in the hierarchical fit of [1] (Section F2.7).

*Rendering on one circle.* A partial of constant amplitude is the vertical projection of a radius vector of fixed length $A$ whose angular speed is the instantaneous frequency. A frequency gradient is therefore an *angular acceleration* on one circle, and an amplitude gradient turns the circle into a spiral (Definition 3). To render a track as a circle curve, split it into sub-supports of length $L_s$ and traverse each at the constant rate that reaches the exact angle at its end. The rotations share the centre and the radius, the angle is continuous, and, as at the interior anchor of a circle (Section F2.4), only the angular speed changes at the joins. For a linear frequency the angle $\theta$ is quadratic in time, so the rendered angle $\hat\theta$ misses it by at most $\gamma L_s^2/8$ inside a sub-support, and

$$
\bigl\lvert A\sin\theta-A\sin\hat\theta\bigr\rvert\le A\lvert\theta-\hat\theta\rvert\le\frac{A\lvert\gamma\rvert L_s^2}8. \tag{21}
$$

Sub-supports of length $L_s\le\sqrt{8\tau_s/(A\lvert\gamma\rvert)}$ therefore render the partial within a tolerance $\tau_s$. The slope jumps at the joins are at most $A\lvert\gamma\rvert L_s$, and by Corollary F3 they set the $\omega^{-2}$ tail of the rendered track. A decoded track is thus a circle curve, built by the same equal-angle traversal as SinWhisky's half-arcs, and FAFT evaluates its spectrum exactly: an encoder can compare the decoded and the source spectra at any frequency.

### 4.4 Why lower frequencies

Four observations support starting with the low bands. Only the first two are established.

1. *Sampling.* The sample rate that a band needs is proportional to its upper edge (Section 4.1).

2. *Geometry scales the same way.* **Proposition 9.** Let a track be band-limited to $\omega_e$, that is, of exponential type $\omega_e$, and bounded by $A$ on the real line. On any support of length $L\le\sqrt{8\tau/A}/\omega_e$, the chord through the end values of the support is within $\tau$ of the track. A curve of lines within $\tau$ of $N$ samples therefore needs at most $\lceil N\omega_e\sqrt{A/(8\tau)}\rceil$ pieces. *Proof.* By Bernstein's inequality [19], $\lvert y''\rvert\le\omega_e^2A$, and linear interpolation over an interval of length $L$ errs by at most $L^2\max\lvert y''\rvert/8$. $\square$ The same $1/\omega_e$ scaling of the admissible support length holds for polynomial pieces of any fixed degree. Isolating a low band therefore shortens a description by pieces in proportion to its edge, as it shortens the sample sequence: band division helps a geometric description by about the same factor as it helps sampling, and any advantage of circles over samples must come from the constant, which is to be measured (Section 6).

3. *Partials.* The cost of the tonal layer scales with the number of partials and with how fast they change, not with the band edge. The low bands of music are expected to hold few partials of long duration, such as bass lines and fundamentals, so that each window adds few breakpoints and thinning removes most of them. This is a hypothesis about the content.

4. *Resolution.* In absolute frequency, low partials are crowded: the harmonics of a 55 Hz note are 1.28 bins apart at 44.1 kHz and $N=1024$, and they form the long chains of Section 2.4. The cost of FAFT is proportional to the number of rotations, not of samples, and by observation 2 an isolated low band can be described by few pieces. A window several times longer, which divides the bin width and loosens the chain, should therefore cost the low band no more than a short window costs the full track; how many rotations the hierarchical fitter actually needs is experiment 5 of Section 6.

### 4.5 Layered coding with integer corrections

The encoder writes three layers, each against the decoded output of the previous ones (closed loop).

1. *Tonal layer.* For each band, the thinned and quantized breakpoints of its partials. The decoder renders them (Section 4.3) into $\hat{\mathcal T}_b$.
2. *Band layer.* For each band, samples of $y_b-\hat{\mathcal T}_b$, taken at the band's own rate with a guard band and quantized with a step $q_b$. The decoder restores them to the source rate by band-limited interpolation. Since this layer is computed from the *decoded* tones, the quantization error of the tonal parameters enters it and is coded, instead of being lost.
3. *Final layer.* At every source sample, $e[n]=x[n]-\hat x_{12}[n]$, where $\hat x_{12}$ is the sum of the decoded layers 1 and 2. If $\lvert e[n]\rvert\le\tau$, nothing is written. Otherwise the position $n$ and the integer $j_n=\operatorname{round}\bigl(e[n]/(2\tau)\bigr)$ are written, and the decoder adds $2\tau j_n$. Only the samples whose difference from the model exceeds the tolerance are written, and they are written as small integers, with a precision far below that of the source.

**Proposition 10 (error contract).** Whatever layers 1 and 2 contain, however they are quantized, and however well the model fits, every decoded sample satisfies $\lvert x[n]-\hat x[n]\rvert\le\tau$. The decoder needs no source samples. For integer PCM, with $\hat x_{12}$ rounded to integers and $\tau=0$, where every nonzero $e[n]$ is written as it is, the reconstruction is lossless.

*Proof.* A written sample has the error $\lvert e[n]-2\tau j_n\rvert\le\tau$ by rounding; an unwritten one has $\lvert e[n]\rvert\le\tau$. The decoder reads only the layers and the positions. For $\tau=0$, every $e[n]$ is an integer and is written exactly. $\square$

The cost of the final layer per source sample is about

$$
R_3\approx H_2(f)+f\bar n\quad\text{bits},\qquad H_2(f)=-f\log_2f-(1-f)\log_2(1-f), \tag{22}
$$

where $f$ is the fraction of samples written, $H_2(f)$ the cost of marking their positions with an ideal entropy coder, and $\bar n$ the mean entropy-coded length of a written integer [20]. The better layers 1 and 2 predict, the smaller $f$ and the integers become.

*A prediction from the residual energy.* With entropy-coded uniform quantization at a step small relative to the signal, the cost per sample is about the differential entropy minus $\log_2$ of the step, which for Gaussian values is $\tfrac12\log_2(2\pi e\sigma^2)$ minus $\log_2$ of the step [21]. Coding the residual of a model that leaves the fraction $E_r/E$ of the energy, instead of the signal, at the same step, therefore saves about

$$
\frac12\log_2\frac E{E_r}\quad\text{bits per sample}, \tag{23}
$$

one bit per 6.02 dB of explained energy [22]. The joint fit gives $E$ and $E_r$ in every window (Proposition F8) before anything is coded, so (23) predicts the gain of the tonal layer window by window. On the synthetic signal of Section F7, whose fit explained a median 99.9992% of the weighted energy, (23) gives about 8.5 bits per sample. This is an optimistic estimate, not a measurement: it ignores the cost of the parameters, assumes Gaussian residuals and fine quantization, and refers to a nearly stationary synthetic signal, while the coded residual is that of the *quantized* parameters.

```text
Algorithm 3  ENCODE(samples x, edges, τ)
  y ← circle curve of x (Section F2.7);  (tonal_b, r_b) ← DIVIDE(y)    ▷ Algorithm 2
  for b = 1 … B:                                         ▷ low bands first
      tracks ← link components over windows (Algorithm F6), with gradients (17)–(20)
      thin breakpoints (Section 4.3); quantize; write
      tonal'_b ← render the decoded tracks                ▷ rotations, (21)
      d_b ← samples of y_b − tonal'_b at the band's rate  ▷ closed loop
      quantize d_b with step q_b; write;  d'_b ← band-limited interpolation
  x' ← Σ_b (tonal'_b + d'_b) at the source samples
  for each n:  e ← x[n] − x'[n]
      if |e| > τ:  write (n, round(e / 2τ))                 ▷ small integers
DECODE: render the tracks, interpolate the band samples, add 2τ·j_n where written
```

### 4.6 What is transmitted

Circles enter this scheme twice, and neither time as transmitted descriptors of the source. The determined curve is the object that FAFT analyses and the residual divider filters, and the decoded tracks are rendered as rotations. What is written is breakpoints, band samples and integer corrections. This is deliberate: the measurements recalled in Section 5 found that local circle descriptors do not pay for themselves when they replace samples one support at a time.

---

## 5. Relation to earlier work

**Parametric coding.** Sinusoidal analysis and synthesis with tracked partials [17] and its deterministic-plus-stochastic extension [23] are the basis of parametric audio coding, including the sines, transients and noise model of [15] and the HILN tools of MPEG-4 [24]. The layered structure of Section 4 follows them. What this paper adds is the continuous-axis machinery that produces the layers: the exact transform of the analysis curve, closed-form joint fits with a residual bound, a banded two-way solution, gradients from transform derivatives, and exact filtering of the curve.

**Dividers.** Complementary crossovers, which add back to the input by construction, are classical in loudspeaker design [2, 3], and subband coding splits a signal into bands sampled at their own rates [4]. Dividers by spectral masking derive from short-time Fourier analysis and synthesis [6]. The component divider of Section 2.3 is neither a filter bank nor a mask: it assigns fitted components.

**Prediction and residuals.** Lossless audio coders transmit a predictor and the integer residual of its rounded prediction [25]. Section 4.5 replaces the linear predictor with the decoded layers and keeps the integer residual as the final, bounded layer.

**Earlier measurements in this project.** The compression lab of this project (`docs/COMPRESSION.md`) coded three one-second excerpts of a Beethoven recording with trees of local circles and lines, residuals and sparse Fourier bases, counting the bytes of complete files. At a maximum error of 16 PCM16 units, circle and line residual coding used 65.29% of the PCM size and saved one byte over line-only coding across the three files; FLAC used 36.08% with zero error; and the searches kept almost no circle descriptors. These results concern circles used as transmitted, per-support predictors of the full band. They are the reason the present scheme transmits tracks instead of circles, and they set the controls against which it must be measured.

---

## 6. Implementation and experiments

### 6.1 First implementation and measurements

The crate now implements, as experiments: the two-way banded fit of Proposition 3 with selected inversion (`Faft::fit_banded`, in `src/faft/banded.rs`) and a per-component Newton refinement of the frequencies (`Faft::fit_refined`); exact analog filtering of circle curves through spiral pieces, Butterworth low-passes, and zero-phase forward–backward passes (`src/analog.rs`); the component divider, equalization and extrusion of Sections 2–3 (`src/bands.rs`); and a first, simplified layered coder (SWL1, since replaced by SWL2 of Section 6.2). Its tests check the banded fit against the dense fit, the filter against Runge–Kutta integration to $10^{-8}$, steady-state and zero-phase responses to $10^{-9}$, completeness of the division to $10^{-9}$, and exact or bounded round trips of every coded stream.

The implementation departs from Sections 2–4 in five stated ways. (i) Tones are selected as the exact maxima of $\lvert\mathcal A\rvert$ below an upper frequency, fitted jointly by the banded solve, pruned by significance and refined by three rounds of the coordinate Newton step, not by the pursuit of [1]: on this recording the tracker of Section F6.7 rescanned in almost every window and cost 100–330 ms per window in development runs, against about 30 ms here. (ii) Frequencies below half a resolution are fitted as DC, a cosine at $\omega=0$. Within a hair of DC the sine atom has almost no norm, and a fit there produced an amplitude of $2.4\cdot10^{6}$ for a slow drift; the overlap-add of the window still reproduced it, but a rendered track did not. (iii) The residual of the divider is filtered as the line curve through its samples. (iv) The coder has no band layer. Tracks store frequency, amplitude and phase at every window centre, with the frequency predicted by the track's own gradient and the phase by integration, and are rendered with the cubic phase of [17] rather than the Hermite gradients of Section 4.3; there is no breakpoint thinning yet. (v) The final layer predicts each block of 4096 samples with a closed-loop polynomial of order 0, 1 or 2 and quantizes integer samples with the odd step $2\tau+1$, which bounds the error by $\tau$ without a separate dead zone.

The measurements use one 5-second excerpt (60–65 s) of the stereo Beethoven recording of Section F7, decoded to PCM16 (882,000 bytes), with windows of $T=2048$ and a circle-curve tolerance of 16. The tones up to 2 kHz explained 90.5% and 91.5% of the weighted energy of the two channels, with 25.7 tones per window. The three bands of a divider with edges at 250 Hz and 2 kHz added back to the source within $3.6\cdot10^{-12}$; their RMS levels were 1066, 2249 and 366. Table 2 lists the coded sizes; every stream was decoded and reproduced the encoder's samples, within its tolerance.

| Tonal band | $\tau$ | Bytes | % of PCM | Tonal layer | Correction layer | Corrections written | Residual RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| none (control) | 0 | 401,975 | 45.58 | — | 401,907 | 98.2% | 2536 |
| up to 2 kHz | 0 | 414,021 | 46.94 | 27,880 | 386,073 | 97.7% | 683 |
| up to 6 kHz | 0 | 392,219 | 44.47 | 45,367 | 346,784 | 96.3% | 536 |
| none (control) | 16 | 154,540 | 17.52 | — | 154,472 | 70.5% | 2536 |
| up to 2 kHz | 16 | 172,419 | 19.55 | 27,880 | 144,471 | 65.7% | 683 |
| up to 6 kHz | 16 | 164,377 | 18.64 | 45,367 | 118,942 | 55.4% | 536 |
| none (control) | 128 | 77,613 | 8.80 | — | 77,545 | 38.8% | 2536 |
| up to 6 kHz | 128 | 87,734 | 9.95 | 45,367 | 42,299 | 18.5% | 536 |

*Table 2. Layered coding of one 5-second stereo excerpt (`docs/experiments/eq-compression-beethoven-20261005.json`). Bytes are complete streams of both channels, headers included. The residual RMS is that of the source minus the rounded tonal track, which the correction layer codes; FLAC coded the same excerpt losslessly in 328,541 bytes (37.25%).*

Three findings follow, from one excerpt. First, the tonal layer removed 11–14 dB of residual energy, but the correction layer shrank far less, because its polynomial predictor already removes most low-frequency content: a second-order difference attenuates a tone at $\omega$ by about $\omega^2$ in amplitude. The low band, where Section 4.4 expected the tracks to pay first, is the band that a sample-domain predictor already handles. Second, tracks up to 6 kHz saved 2.4% losslessly against the control, while tracks up to 2 kHz cost 3.0% more; with a tolerance the tracks never paid, because the correction layer then writes few samples and the tracks cost the same. Third, both the control and the layered streams are far above FLAC losslessly, whose higher-order linear prediction is the stronger baseline. A layered coder must therefore be measured against linear prediction, not against raw samples, and the tonal layer must pay for its parameters in the band where that prediction fails.

### 6.2 Second coder: linear prediction, gradient groups and SinWhisky anchors

The findings of Section 6.1 led to a second coder, SWL2, specified byte by byte in `docs/SWL2.md`. (i) The correction layer predicts with LPC up to order 32: Levinson–Durbin on the Tukey-windowed autocorrelation of each block, with the variance of the quantization noise of the reconstructed history added to its zero lag in lossy streams, and 15-bit integer coefficients, so that the decoder of this layer is exact on any platform. It takes the earlier layers' prediction as side information: each block codes `x - round(g P)` with LPC on the same quantity, choosing the gain g among 0, 1/4, …, 1 and the order by the actual coded size, so that a layer cannot enlarge a block's residual. (ii) The tonal layer is a list of frequency-gradient groups. Track segments with quadratic phase, that is, a constant frequency gradient, are grouped by start and duration with a shared volume function (constant, linear or one exponential ratio), and adjacent segments of a track are cross-faded over one hop. The chunk header declares the bit width of every field, the smallest that holds its largest value. (iii) A SinWhisky chunk describes a low band, or everything the gradients left, by anchors at dynamic positions, rebuilt by the circles through anchor triplets, with an angle quadratic in time, and cross-faded between neighbouring circles; the anchors are placed greedily until every interval is within a tolerance. (iv) `encode_auto` chooses the stereo mode (independent, left/side, side/right or mid/side, losslessly only), and for each channel whether gradients pay (band, precision, segment tolerance), whether anchors pay (low band or everything, tolerance), and the block size, by the exact size of the complete stream. (v) A parametric mode drops the correction layer; its error is measured as a signal-to-noise ratio, not bounded. (vi) In version 3 of the stream, a cascade of three sign-sign LMS stages of 1024, 256 and 16 taps follows the LPC, as in Monkey's Audio. Each stage predicts what the earlier ones left from its own reconstructed history and adapts after every sample in integer arithmetic, so this layer stays exact on any platform. The stages always adapt; each block codes how many of them it adds (0 to 3), chosen by exact coded size on the two best LPC candidates.

Table 3 lists three 5-second excerpts of the same recording, with the full search: tones analysed up to 6 kHz, about 50 s per stream and tolerance on the laptop of Section F7.

| Excerpt | Lossless SWL2 (FLAC) | $\tau=16$: bytes, SNR | $\tau=128$: bytes, SNR | Parametric: bytes, SNR | Layers kept |
|---|---:|---:|---:|---:|---|
| 60 s | 303,109, 34.37% (328,541, 37.25%) | 125,756, 48.5 dB | 62,856, 30.8 dB | 158,755, 29.0 dB | none |
| 180 s | 204,765, 23.22% (230,035, 26.08%) | 62,305, 34.0 dB | 22,580, 16.8 dB | 27,168, 14.3 dB | gradients at $\tau=16$ and 128 |
| 300 s | 380,505, 43.14% (415,142, 47.07%) | 181,763, 51.4 dB | 94,016, 33.5 dB | 263,143, 32.3 dB | none |

*Table 3. SWL2 with the LPC-only correction layer (version 2) on three excerpts of 882,000 PCM bytes each (`docs/experiments/swl2-beethoven-20261005.json`). Every stream was decoded and reproduced the encoder's samples within its tolerance. The parametric stream holds gradient groups up to 6 kHz at precision 16 and anchors at tolerance 256. The automatic search kept independent stereo in every case.*

Four findings follow. First, the gain is in the correction layer. Losslessly, SWL2 is 7.7%, 11.0% and 8.3% smaller than FLAC on the three excerpts, and on the 60 s excerpt it is 24.6%, 18.6% and 19.0% smaller than the first coder at tolerances 0, 16 and 128. Second, with linear prediction the gradient and anchor layers do not pay for loud passages. In a development run on one channel of the 60 s excerpt, the finest gradient layer lowered the residual by 12.8 dB but the correction layer shrank by only 4.5% (6.9 kB) for a layer of 74.9 kB: a predictor of order 20 already removes the energy of stationary partials, which is the same energy the tones describe. A low-band anchor layer saved nothing, and the correction blocks gave it gain 0. Third, on the quiet, tonal 180 s excerpt the search kept the gradients: at $\tau=128$ they made the stream 16.9% smaller (27,176 to 22,580 bytes) with a slightly higher SNR, and at $\tau=16$ 1.8% smaller. With coarse quantization the predictor works from a noisy history, whereas the gradients are side information that the quantization does not touch; a layer of about 4.7 kB left the correction layer to write only 4.5% of the samples. Fourth, the parametric streams describe these passages with fewer samples, but at every SNR measured they need more bytes than the correction layer at a comparable SNR (for example 158,755 bytes at 29.0 dB against 62,856 bytes at 30.8 dB). Anchors alone, in another development run, needed about 120 kB per channel for 41 dB, where the correction layer needs about 40 kB per channel for 37 dB.

Table 4 repeats the measurement with the LMS cascade (vi). Without the cascade, version 3 reproduces the byte counts of the LPC-only coder exactly (checked on the 60 s excerpt and on the fast search of the 180 s excerpt).

| Excerpt | Lossless SWL2 (vs FLAC) | $\tau=16$: bytes, SNR | $\tau=128$: bytes, SNR | Saved against Table 3 ($\tau$ = 0, 16, 128) | Layers kept |
|---|---:|---:|---:|---:|---|
| 60 s | 288,521, 32.71% (−12.2%) | 114,795, 48.5 dB | 55,042, 30.9 dB | 4.8%, 8.7%, 12.4% | none |
| 180 s | 199,805, 22.65% (−13.1%) | 58,616, 34.0 dB | 17,330, 17.4 dB | 2.4%, 5.9%, 23.3% | gradients: one channel at $\tau=16$, both at 128 |
| 300 s | 357,851, 40.57% (−13.8%) | 164,662, 51.4 dB | 82,888, 33.5 dB | 6.0%, 9.4%, 11.8% | none |

*Table 4. SWL2 version 3, with the LMS cascade after the LPC, on the excerpts of Table 3 and with the same full search (`docs/experiments/swl2-nlms-beethoven-20261005.json`). Every stream was decoded and reproduced the encoder's samples within its tolerance. The parametric streams are unchanged.*

The cascade makes lossless SWL2 12.2–13.8% smaller than FLAC, against 7.7–11.0% with LPC alone, at the same SNR when lossy. The blocks used 1.0 to 2.9 stages on average, more when lossless than at $\tau=128$. The saving grows with the tolerance on the two loud excerpts (4.8–6.0% lossless, 11.8–12.4% at $\tau=128$) and is largest on the quiet 180 s excerpt at $\tau=128$ (23.3%), where the search now keeps the gradients on both channels. The adaptive stages reach 1024 samples back, beyond the order-32 LPC. The price is encoding time: on the fast search of the 180 s excerpt, 5.2 s against 3.2 s per lossless stream and about 5 s against 1.7–1.8 s per lossy one, because a lossy trial must run the stages again for each choice. In development runs on the 60 s excerpt (fast search), stages of 256, 32 and 16 taps saved 0.8 points less losslessly and 2.1 points less at $\tau=128$; a first stage of 1280 taps changed the size by at most 0.2%, and one of 2048 taps was larger at both tolerances.

The parametric and geometric layers therefore help where waveform prediction is weak (coarse tolerances, quiet tonal passages), and Section 4.4's expectation that low bands would pay first is not supported: the low band is what linear prediction captures best. Whether gradient groups and SinWhisky anchors pay under an error criterion that does not demand the waveform, as perceptual coders use, cannot be decided from byte counts and SNR; it needs listening tests.

### 6.3 Remaining experiments

The experiments below have not been run, or have been run only as far as Section 6.1 states. A claim follows only from their measured outcomes.

| # | Experiment | Measured | Controls |
|---|---|---|---|
| 1 | Divider selectivity: synthetic tones at known distances from edges, with noise | Tonal and residual crosstalk per band; completeness to rounding | Spectral masking (rectangular, Hann); Linkwitz–Riley crossover on samples |
| 2 | Two-way banded fit | Agreement with the dense joint fit; cost against chain length; amplitude change against reach | Clustered solve of [1] (`pursue`) |
| 3 | Exact analog filtering of circle curves | Agreement with dense numerical integration; cost per pole and rotation | Digital filter applied to zoomed samples |
| 4 | Gradient tones | Bias and spread of $\hat\gamma$ and $\hat\varrho$ against $\gamma T^2$ and noise; residual energy against stationary tones | Stationary pursuit; numerical nonlinear chirp fit |
| 5 | Low-band geometry | Rotations per second of determined curves against band edge, on the windows of Section F7 | Samples at the band's Nyquist rate |
| 6 | Bytes | Complete files (all layers, headers, positions) at $\tau=0$, 16 and 128 PCM16 units on the excerpts of Section 5; every decoded sample checked against $\tau$ | FLAC; the methods of the compression lab; PCM with zlib |
| 7 | Bands with their own windows | Convergence of backfitting with per-band windows | Single-window joint fit |

*Table 4. Remaining measurements.*

Listening is not on this list. Error bounds and byte counts do not establish perceptual quality, and any listening comparison needs its own blind protocol.

---

## 7. Discussion and limitations

**The divider is the fit.** The selectivity of the tonal part is that of the joint fit. What the fit does not explain goes to the residual divider, which is an ordinary linear filter with a transition band, even if it is applied exactly. Noise and transients are divided no better than by a crossover.

**Time resolution is the hop.** Tones enter and leave a track through the overlap of windows, and an onset inside a window is represented by tones only approximately: in Section F7 the tracker spent short-lived components on an onset. For coding, the closed loop keeps the error bound, but the cost moves to the final layer.

**Edges, glides and the axis.** A partial that crosses an edge changes track, and smooth weights only spread the change. The content of the curve above the source Nyquist frequency is produced by the circles (Section F8), so dividing it is meaningful for the curve only.

**Joins.** Exact filtering applies to the curve as fitted, including the tolerance-bounded value jumps at the joins of rounded circles, which excite the filter's step response.

**Gradients.** The gradient tone is a first-order local model. Its errors grow with $\gamma T^2$, and the nonlinear chirp has no elementary closed form.

**Bands with their own windows.** The convergence of backfitting across different windows is open.

**Compression.** Nothing in Section 4 has been measured. Earlier circle descriptors did not pay for themselves (Section 5), the parameters of the tonal layer cost bits that (23) ignores, and entropy coding, stereo coupling and the costs of seeking and streaming are not addressed.

---

## 8. Conclusion

Dividing a track by frequency is the problem of Section F6 in another form: the energy near an edge cannot be assigned until it is explained, and a masked spectrum assigns each tone's leakage to the wrong band. The joint fit of [1] divides components instead, which makes the division complete by construction and leaves the tonal part of every band free of the other bands' tones. In frequency order the fit is banded, and its exact solution is one pass from low to high frequencies, which detects and accounts for the energy, and one pass back, which corrects the amplitudes. What no tone explains is divided by analog filters applied exactly to the circle curve, through the same rotate-and-add chain as FAFT with complex rates, turning circles into spirals whose transform stays in closed form. Equalization and extrusion follow from the divider. For compression, the tonal part of a band becomes tracks of frequency gradients, estimated in closed form and rendered as circles with stepwise angular speed, and a final layer of small integers keeps every sample within a stated tolerance. Whether this representation needs fewer bytes than samples, starting from the low bands, is the question that the experiments of Section 6 must answer.

---

## References

1. R. Cecchini, "The Angular Fourier Transform for obtaining a continuous spectrum," manuscript, 2026. Included in the repository as `docs/PAPER_FAFT.md`.
2. S. H. Linkwitz, "Active crossover networks for noncoincident drivers," *Journal of the Audio Engineering Society* 24 (1976) 2–8.
3. S. P. Lipshitz and J. Vanderkooy, "A family of linear-phase crossover networks of high slope derived by time delay," *Journal of the Audio Engineering Society* 31 (1983) 2–20.
4. R. E. Crochiere, S. A. Webber and J. L. Flanagan, "Digital coding of speech in sub-bands," *Bell System Technical Journal* 55 (1976) 1069–1085.
5. A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010.
6. J. B. Allen, "Short term spectral analysis, synthesis, and modification by discrete Fourier transform," *IEEE Transactions on Acoustics, Speech, and Signal Processing* 25 (1977) 235–238.
7. G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013.
8. K. Takahashi, J. Fagan and M.-S. Chen, "Formation of a sparse bus impedance matrix and its application to short circuit study," *Proceedings of the 8th Power Industry Computer Applications Conference*, 1973, 63–69.
9. A. George and J. W. H. Liu, *Computer Solution of Large Sparse Positive Definite Systems*, Prentice-Hall, 1981.
10. T. J. Hastie and R. J. Tibshirani, *Generalized Additive Models*, Chapman and Hall, 1990.
11. Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
12. R. Bristow-Johnson, "Cookbook formulae for audio equalizer biquad filter coefficients," online note, 2005.
13. G. F. Franklin, J. D. Powell and M. L. Workman, *Digital Control of Dynamic Systems*, 3rd ed., Addison-Wesley, 1998.
14. J. Kormylo and V. Jain, "Two-pass recursive digital filter with zero phase shift," *IEEE Transactions on Acoustics, Speech, and Signal Processing* 22 (1974) 384–387.
15. S. N. Levine, *Audio Representations for Data Compression and Compressed Domain Processing*, PhD thesis, Stanford University, 1998.
16. S. Peleg and B. Porat, "Estimation and classification of polynomial-phase signals," *IEEE Transactions on Information Theory* 37 (1991) 422–430.
17. R. J. McAulay and T. F. Quatieri, "Speech analysis/synthesis based on a sinusoidal representation," *IEEE Transactions on Acoustics, Speech, and Signal Processing* 34 (1986) 744–754.
18. D. H. Douglas and T. K. Peucker, "Algorithms for the reduction of the number of points required to represent a digitized line or its caricature," *Cartographica* 10 (1973) 112–122.
19. R. P. Boas, *Entire Functions*, Academic Press, 1954.
20. T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006.
21. R. M. Gray and D. L. Neuhoff, "Quantization," *IEEE Transactions on Information Theory* 44 (1998) 2325–2383.
22. N. S. Jayant and P. Noll, *Digital Coding of Waveforms: Principles and Applications to Speech and Video*, Prentice-Hall, 1984.
23. X. Serra and J. O. Smith III, "Spectral modeling synthesis: a sound analysis/synthesis system based on a deterministic plus stochastic decomposition," *Computer Music Journal* 14(4) (1990) 12–24.
24. H. Purnhagen and N. Meine, "HILN — the MPEG-4 parametric audio coding tools," *Proceedings of the IEEE International Symposium on Circuits and Systems*, 2000, vol. 3, 201–204.
25. T. Robinson, "SHORTEN: simple lossless and near-lossless waveform compression," Technical Report CUED/F-INFENG/TR.156, Cambridge University Engineering Department, 1994.

### Reproducibility notes

The methods are defined by Definitions 1–3, Propositions 1–10 and Algorithms 1–3, on top of the definitions of [1]. `tools/eq_compression_checks.py` (Python standard library, deterministic, no audio) computes Table 1 and the figures quoted in Sections 2.4 and 4.2, and checks Proposition 3 (the explained energy from the ascending pass, and a banded factor without fill-in), Proposition 5 (filter states carried across three rotations against Runge–Kutta integration), Proposition 6 (the spiral transform (13) and its endpoint form (14) against Gauss–Legendre quadrature) and Proposition 8 (the normal equations (20) against a direct least-squares fit, images included, and the frequency gradient of a chirp). The implementation of Section 6.1 is in `src/faft/banded.rs`, `src/analog.rs`, `src/bands.rs` and `src/layered/`, tested in `tests/bands.rs` and `tests/layered.rs`; `examples/eq_audio.rs`, driven by `test_music.sh` (`quick`, `divide`, `eq`, `extrude`, `compress`), produces the aggregate reports behind Tables 2, 3 and 4; the SWL2 byte layout is `docs/SWL2.md`. Section 6.3 lists the measurements that remain.
