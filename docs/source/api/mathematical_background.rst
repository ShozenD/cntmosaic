Mathematical Background
======================

This section provides mathematical details for the models and methods implemented in Contact Mosaic.

Contact Matrix Theory
----------------------

Definitions
^^^^^^^^^^^

A **contact matrix** :math:`C = [c_{ij}]` represents the average number of contacts between individuals in age group :math:`i` and age group :math:`j`.

The **contact intensity** :math:`M = [m_{ij}]` is defined as:

.. math::

   m_{ij} = \frac{c_{ij}}{N_i}

where :math:`N_i` is the number of participants in age group :math:`i`.

The **contact rate** :math:`\omega = [\omega_{ij}]` incorporates population structure:

.. math::

   \omega_{ij} = \frac{m_{ij} \cdot N_i}{P_i}

where :math:`P_i` is the population size in age group :math:`i`.

Reciprocity
^^^^^^^^^^^

The **reciprocity condition** ensures that the total number of contacts from group :math:`i` to :math:`j` equals contacts from :math:`j` to :math:`i`:

.. math::

   m_{ij} \cdot P_i = m_{ji} \cdot P_j

This is fundamental to constructing symmetric contact matrices used in epidemic models.

Bayesian Rate Consistency Model
--------------------------------

Model Structure
^^^^^^^^^^^^^^^

The BRC model assumes contact counts follow a negative binomial distribution:

.. math::

   y_{ijk} \sim \text{NegBin}(\mu_{ijk}, \phi)

where:

- :math:`y_{ijk}` is the observed contact count for participant :math:`k` of age :math:`i` contacting age :math:`j`
- :math:`\mu_{ijk}` is the expected contact count
- :math:`\phi` is the dispersion parameter

The expected count is modeled as:

.. math::

   \log(\mu_{ijk}) = \log(N_k) + \log(S_k) + \log(\lambda_{ij}) + \log(P_j) + h(r_k)

where:

- :math:`N_k` is the number of contacts reported by participant :math:`k`
- :math:`S_k` is a survey-specific offset
- :math:`\lambda_{ij}` is the contact rate
- :math:`h(r_k)` is a Hill function for repeat contacts

Rate Consistency
^^^^^^^^^^^^^^^^

The **rate consistency constraint** ensures:

.. math::

   \lambda_{ij} = \lambda_{ji}

This is enforced through the use of symmetric priors on the log-rate field.

Spatial Priors
--------------

Intrinsic Gaussian Markov Random Fields (IGMRF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An IGMRF on a 2D lattice has precision matrix:

.. math::

   Q = \tau \cdot (D - W)

where:

- :math:`\tau` is the precision parameter
- :math:`D` is the diagonal matrix of neighbor counts
- :math:`W` is the adjacency matrix

For a :math:`d`-th order difference penalty:

.. math::

   Q = \tau \cdot \Delta_d^T \Delta_d

where :math:`\Delta_d` is the :math:`d`-th order difference operator.

Log-probability:

.. math::

   \log p(x | \tau) = \frac{1}{2} \log |Q^*| - \frac{1}{2} x^T Q x + \text{const}

where :math:`Q^*` is the generalized inverse excluding the null space.

Hilbert Space Gaussian Process (HSGP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HSGP approximates a Gaussian process using spectral representation:

.. math::

   f(x) \approx \sum_{m=1}^{M} \beta_m \phi_m(x)

where :math:`\phi_m` are eigenfunctions of the Laplacian operator and :math:`\beta_m \sim N(0, \lambda_m)` with eigenvalues :math:`\lambda_m`.

For a squared exponential covariance:

.. math::

   k(x, x') = \sigma^2 \exp\left(-\frac{||x - x'||^2}{2\ell^2}\right)

the eigenvalues decay as:

.. math::

   \lambda_m \approx \sigma^2 \exp\left(-\frac{\ell^2 \pi^2 m^2}{2L^2}\right)

Penalized B-Splines (P-Splines)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

P-splines use B-spline basis functions with a difference penalty:

.. math::

   f(x) = \sum_{k=1}^{K} \beta_k B_k(x)

The penalty on coefficients is:

.. math::

   P(\beta | \tau) = \exp\left(-\frac{\tau}{2} \sum_{k} (\Delta^d \beta_k)^2\right)

where :math:`\Delta^d` is the :math:`d`-th order difference operator.

For 2D tensor product splines:

.. math::

   f(x, y) = \sum_{i,j} \beta_{ij} B_i(x) B_j(y)

with IGMRF prior on the vectorized coefficients :math:`\text{vec}(\beta)`.

Compositional Data Transformations
-----------------------------------

Contact matrices often represent probabilities across age groups, requiring transformations to unconstrained space.

Additive Log-Ratio (ALR)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \text{ALR}(p) = \left[\log\frac{p_1}{p_K}, \ldots, \log\frac{p_{K-1}}{p_K}\right]

Inverse:

.. math::

   \text{ALR}^{-1}(y) = \frac{1}{1 + \sum_{i=1}^{K-1} e^{y_i}} \left[e^{y_1}, \ldots, e^{y_{K-1}}, 1\right]

Centered Log-Ratio (CLR)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \text{CLR}(p) = \left[\log p_i - \frac{1}{K}\sum_{j=1}^{K} \log p_j\right]_{i=1}^K

Isometric Log-Ratio (ILR)
^^^^^^^^^^^^^^^^^^^^^^^^^^

ILR uses an orthonormal basis :math:`V` of the simplex:

.. math::

   \text{ILR}(p) = V^T \text{CLR}(p)

with dimension :math:`K-1`.

Inference Methods
-----------------

Hamiltonian Monte Carlo (HMC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HMC uses Hamiltonian dynamics to propose samples:

.. math::

   H(\theta, \rho) = -\log p(\theta | y) + \frac{1}{2}\rho^T M^{-1} \rho

where :math:`\rho` is momentum and :math:`M` is the mass matrix.

The No-U-Turn Sampler (NUTS) automatically tunes the trajectory length.

Stochastic Variational Inference (SVI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SVI approximates the posterior with a variational distribution :math:`q(\theta; \lambda)`:

.. math::

   \min_\lambda \text{KL}(q(\theta; \lambda) || p(\theta | y))

Using the reparameterization gradient:

.. math::

   \nabla_\lambda \text{ELBO} = \mathbb{E}_{z \sim q_0}[\nabla_\lambda \log q(g(z; \lambda); \lambda) - \nabla_\lambda \log p(g(z; \lambda), y)]

Optimized using Adam or other stochastic optimizers.

References
----------

1. Mossong, J., et al. (2008). Social contacts and mixing patterns relevant to the spread of infectious diseases. *PLoS Medicine*, 5(3), e74.

2. Prem, K., et al. (2017). Projecting social contact matrices in 152 countries using contact surveys and demographic data. *PLoS Computational Biology*, 13(9), e1005697.

3. Dan, S., et al. (2023). Estimating fine age structure and time trends in human contact patterns from coarse contact data: The Bayesian rate consistency model. *PLoS Computational Biology*.

4. Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and Applications*. Chapman & Hall/CRC.

5. Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89-121.

6. Riutort-Mayol, G., et al. (2023). Practical Hilbert space approximate Bayesian Gaussian processes for probabilistic programming. *Statistics and Computing*, 33(1), 17.
