# Depth-relative trust for carried and drawn seed prompts

Status: definitions, Oct 25 2026, for the v7 prompt encoding (`docs/adjacent_circle_seed_plan.md`) and for later
publication. Pixel quantities are in mosaic pixels at the request's downsample unless stated.

## 1. Notation

- $S \subset \mathbb{R}^2$: a seed shape (a carried polygon, a drawn or carried circle), as the set of its interior
  pixel centres; $\partial S$ its boundary.
- $d(p) = \operatorname{dist}(p, \partial S)$ for $p \in S$: the Euclidean distance transform (depth).
- $D = \max_{p \in S} d(p)$: the shape's maximum depth (the radius of its largest inscribed circle).
- $T$: the target polygon (training only), $P$ a polygon, $B$ a disc.

## 2. Trust profile

With $a$ = `full_frac` (0.5) and $b$ = `zero_frac` = `circle_diameter_scale` (0.95), $0 \le a < b \le 1$:

$$
w(\rho) =
\begin{cases}
1 & \rho \le a \\
\tfrac{1}{2}\left(1 + \cos\left(\pi \, \dfrac{\rho - a}{b - a}\right)\right) & a < \rho < b \\
0 & \rho \ge b
\end{cases}
$$

$w$ is continuous, non-increasing, and $C^1$ at $\rho = a$ and $\rho = b$.

## 3. Global depth coordinate

$$
\rho_g(p) = 1 - \frac{d(p)}{D}, \qquad \rho_g \in [0, 1]
$$

$\rho_g = 0$ at the deepest point, $1$ on the boundary. The fully trusted core is $\{ d \ge (1-a) D \}$, the part deeper
than half the maximum depth at $a = 0.5$; the prompt is zero on the outer band $\{ d \le (1-b) D \}$, the outer 5% of
the maximum depth at $b = 0.95$.

### Circles are the special case

For a disc $B(c, R)$: $d(p) = R - \lVert p - c \rVert$ and $D = R$, so

$$
\rho_g(p) = \frac{\lVert p - c \rVert}{R}.
$$

The depth-relative trust of a disc is the radius-relative circle trust (`circle_trust()`, v6 circle annotations). Its
support $\{ w > 0 \} = \{ \lVert p - c \rVert < bR \}$ is the disc of diameter $b \cdot 2R$: $b$ is exactly the circle
diameter scale. The excluded band is a fraction $1 - b$ of the *depth*; as area it is $1 - b^2$ of the disc (9.75% at
$b = 0.95$). Depth is the consistent measure: it is what makes the polygon rule reduce to the circle rule.

## 4. Local depth coordinate (branching shapes)

A global $D$ lets a large body erase a thin branch: a branch of half-width $h \ll D$ has $\rho_g > 1 - h/D$ everywhere,
so it falls in the zero band when $h < (1-b) D$. Normalising by local thickness keeps each branch's own core.

Local radius (half the local thickness of Hildebrand and Rüegsegger, 1997):

$$
D_\ell(p) = \max \{\, r \;:\; \exists\, c,\ p \in B(c, r) \subseteq S \,\}
$$

the radius of the largest inscribed disc that contains $p$. Then

$$
\rho_\ell(p) = 1 - \frac{d(p)}{D_\ell(p)}, \qquad \rho_\ell \in [0, 1].
$$

Properties: $d(p) \le D_\ell(p) \le D$, and $D_\ell(p) = D$ wherever $p$ lies in a largest inscribed disc of $S$.

- For a disc, every $p$ lies in $B(c, R)$ itself, so $D_\ell \equiv R$ and $\rho_\ell = \rho_g = \lVert p - c \rVert / R$:
  circles are unchanged.
- For a band of half-width $h$ away from any wider part, $D_\ell = h$ and $\rho_\ell = 1 - d/h$: the branch has its own
  core $\{ d \ge (1-a) h \}$.
- Near a junction, branch pixels inside a large body disc take that disc's radius, so they read as the body's outer
  zone, and the body and branch cores stay separate.

Computation: $d$ by the exact Euclidean distance transform; $D_\ell$ by the local-thickness algorithm (for each medial
axis point $c$ with radius $r = d(c)$, paint $r$ into $B(c, r)$, keeping the maximum per pixel; restricting $c$ to the
medial axis gives the same result as all centres).

## 5. Islands

With $\rho = \rho_\ell$ (or $\rho_g$):

- core $K = \{ p \in S : \rho(p) \le a \}$, with connected components $K_1, \dots, K_n$ (8-connectivity);
- falling zone $F = \{ p \in S : a < \rho(p) < b \}$; each $p \in F$ is assigned to the core component nearest to it
  along paths inside $S$ (geodesic distance), so a zone is never shared;
- island $I_i = K_i \cup F_i$. Components with $|K_i| < A_{\min}$ (outline noise, spurs) are dropped with their zone.

The prompt of island $i$ is

$$
m_i(p) = \mathrm{SEED\_LOGIT} \cdot w(\rho(p)) \quad (p \in I_i), \qquad m_i(p) = 0 \text{ otherwise},
$$

combined with the negative context under the overlap rule (a pixel claimed by a positive and a negative region is 0)
and reduced to SAM2's $256 \times 256$ `mask_input` by area averaging.

## 6. Which islands prompt what

Weighted overlap of island $i$ with a polygon $P$:

$$
o_i(P) = \frac{\sum_{p \in I_i \cap P} w(\rho(p))}{\sum_{p \in I_i} w(\rho(p))}
$$

(weighted by trust, so the fully trusted core counts most; with $w \equiv 1$ it is the plain area share).

- **Training**, target $T$: the prompt is the union of the islands with $o_i(T) \ge \theta$ ($\theta$ = 0.60,
  `overlap_min`). If none qualifies the carried shape gives no prompt for $T$ and the variant is drawn again. When a
  carried polygon is linked to several targets (a process branching between sections), each target is prompted by the
  islands that lie on it.
- **Production**: every island is its own prompt and returns its own mask; overlapping results are merged or
  de-duplicated by the client. A branching process therefore comes back as one mask per branch.
- The target mask itself is never part of a prompt (that would teach copying, and production has no target mask).

## 7. Carrying a shape to an adjacent section

Section-to-section map: circle or polygon section $s$ to target section $t$,

$$
\Phi = T_t^{-1} \circ C \circ T_s, \qquad C(v) = v + \alpha f_s(v) + \beta f_t(v)
$$

with $T_s, T_t$ the sections' slice-to-volume (stos) transforms, $f_s, f_t$ the published residual correction fields
(volume nm), and $(\alpha, \beta)$ the production convention.

Circle $(c, R)$, five-point mean scale with $e_1..e_4$ the unit axis vectors and $M$ the rim points that map (all four
are required at export):

$$
c' = \Phi(c), \qquad
R' = R \cdot \frac{1}{|M|} \sum_{k \in M} \frac{\lVert \Phi(c + R e_k) - \Phi(c) \rVert}{R}
$$

Polygon: every ring vertex mapped by $\Phi$.

## 8. Overlap of a carried shape with the target

For export and the trainer's filter, unweighted, on the whole cell:

$$
o(X, T) = \frac{|X \cap T|}{|X|}
$$

for the carried disc $X = B(c', R')$ at its true carried radius, or the carried polygon $X$; holes of $T$ excluded.
Kept when $o \ge \theta$ = 0.60.

## 9. Prompt size against object size

Size ratio of a seed to the object it should select:

$$
s = \frac{D_{\text{prompt}}}{D_T}
$$

with $D_{\text{prompt}} = R'$ for a circle and $D$ of the carried polygon (or of the selected island) for a polygon,
and $D_T$ the target's maximum depth. $s > 1$ is an oversized prompt: an annotator's oversized circle, or a carry that
landed on a thinner part. Reported per family and volume in validation and in the export report, together with the
**wrong-object** rate: results whose core recall on $T$ is low while they cover most of another annotated profile under
the prompt. In production, where $T$ is unknown, the returned mask $Y$ stands in: $\hat{s} = D_{\text{prompt}} / D_Y$
flags a likely oversized circle.

## 10. Reference: edge confidence (v6 seeds)

For comparison, the v6 seed encoding scales by distance from the edge alone:

$$
c(d) = 1 - e^{-d / \tau_{\text{eff}}}, \qquad \tau_{\text{eff}} = \max(\tau_{\min}, \min(\tau, \kappa D))
$$

with $\tau$ = 8 px, $\kappa$ = 0.25, $\tau_{\min}$ = 0.5 px. It trusts nearly the whole shape at full strength; the
depth-relative trust trusts only the inner part relative to the shape's own (local) depth, which is what makes it
robust to oversized circles and drifted carries.

## Reference

Hildebrand T., Rüegsegger P. (1997). A new method for the model-independent assessment of thickness in
three-dimensional images. *Journal of Microscopy* 185(1), 67-75.
