"""
Continuous-Time Neural Stochastic Differential Equation (Neural SDE)
====================================================================

Implements a research-grade Neural SDE using Ito calculus with the
torchsde differential equation solver. The model learns continuous-time
drift (mu) and diffusion (sigma) dynamics conditioned on FinBERT
financial sentiment embeddings.

Architecture:
    1. Sentiment Projector: 768-dim FinBERT CLS embedding -> 32-dim compact representation.
    2. State Encoder: Technical indicators + projected sentiment -> initial latent state y0.
    3. SDE Dynamics: Continuous-time drift f(t, X_t) and diffusion g(t, X_t)
       solved via Euler-Maruyama integration through torchsde.sdeint().
    4. Prediction Decoders: Terminal SDE state y_T decoded to:
       - Gaussian parameters (mu, sigma^2) for price change (NLL loss)
       - Direction probability P(UP) (BCE loss)

Mathematical Foundation:
    dX_t = f(t, X_t) dt + g(t, X_t) dW_t
    where W_t is a standard Wiener process (Brownian motion),
    f is the neural drift capturing expected returns,
    g is the neural diffusion capturing stochastic volatility.
"""

import torch
import torch.nn as nn
import torchsde


class SDEDynamics(nn.Module):
    """
    Ito SDE dynamics for continuous-time financial modeling.

    The drift and diffusion functions are parameterized by neural networks,
    allowing the model to learn complex, nonlinear market dynamics directly
    from data while respecting the mathematical structure of stochastic
    differential equations.
    """
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim

        # f(t, X_t): Neural drift network — deterministic trend component
        self.drift_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # g(t, X_t): Neural diffusion network — stochastic volatility component
        self.diffusion_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid()   # Bound diffusion output to (0, 1)
        )

        # Learnable volatility scale factor
        self.sigma_scale = nn.Parameter(torch.tensor(0.1))

    def f(self, t, y):
        """Drift: f(t, X_t) — expected return dynamics."""
        return self.drift_net(y)

    def g(self, t, y):
        """Diffusion: g(t, X_t) — stochastic volatility dynamics."""
        return self.diffusion_net(y) * self.sigma_scale


class ContinuousNeuralSDE(nn.Module):
    """
    Full continuous-time Neural SDE for sentiment-conditioned stock prediction.

    Pipeline:
        [Technical Features, FinBERT Sentiment]
            -> Encoder -> y0 (initial latent state)
            -> torchsde.sdeint(f, g, y0, [0, T])
            -> y_T (terminal state)
            -> Price Decoder  -> (mu_price, sigma_price)
            -> Direction Decoder -> P(UP)
    """

    def __init__(self, technical_dim, sentiment_dim=768, state_dim=16,
                 hidden_dim=32, dropout_rate=0.3):
        super().__init__()

        self.state_dim = state_dim

        # Sentiment projection: 768-dim FinBERT -> compact representation
        self.sentiment_proj = nn.Sequential(
            nn.Linear(sentiment_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32)
        )

        # State encoder: technical features + projected sentiment -> y0
        self.encoder = nn.Sequential(
            nn.Linear(technical_dim + 32, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, state_dim)
        )

        # Continuous-time SDE dynamics
        self.sde_func = SDEDynamics(state_dim, hidden_dim)

        # Price change decoder: y_T -> Gaussian(mu, sigma^2)
        self.price_decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)    # outputs [mu, log_sigma]
        )

        # Direction decoder: y_T -> P(UP)
        self.direction_decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_technical, x_sentiment, dt=1.0, n_steps=10):
        """
        Forward pass: encode -> solve SDE -> decode predictions.

        Args:
            x_technical:  [batch, technical_dim] scaled technical features
            x_sentiment:  [batch, sentiment_dim] FinBERT CLS embeddings
            dt:           integration time horizon (1.0 = one trading day)
            n_steps:      Euler-Maruyama solver discretisation steps

        Returns:
            mu:             [batch, 1]  predicted expected price change
            sigma:          [batch, 1]  predicted stochastic volatility
            direction_prob: [batch, 1]  probability of UP
        """
        # Project sentiment embeddings
        sent_proj = self.sentiment_proj(x_sentiment)

        # Encode initial SDE state
        combined = torch.cat([x_technical, sent_proj], dim=-1)
        y0 = self.encoder(combined)

        # Solve SDE forward in continuous time via Euler-Maruyama
        ts = torch.linspace(0.0, dt, n_steps + 1, device=y0.device)
        # torchsde.sdeint returns tensor of shape [n_steps+1, batch, state_dim]
        ys = torchsde.sdeint(self.sde_func, y0, ts, method='euler')

        # Terminal state at time T
        y_T = ys[-1]  # [batch, state_dim]

        # Decode price change distribution N(mu, sigma^2)
        price_params = self.price_decoder(y_T)
        mu = price_params[:, 0:1]
        log_sigma = price_params[:, 1:2]
        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=10.0)

        # Decode direction probability
        direction_prob = self.direction_decoder(y_T)

        return mu, sigma, direction_prob


def continuous_sde_loss(mu, true_change, sigma, direction_prob, true_direction):
    """
    Combined loss for the continuous-time Neural SDE.

    1. Negative Log-Likelihood of Gaussian transitions:
       -log N(true_change | mu, sigma^2)  =  0.5 * log(sigma^2) + (x - mu)^2 / (2 sigma^2)

    2. Binary Cross-Entropy for directional classification.
    """
    # NLL of Gaussian transition density
    variance = sigma ** 2 + 1e-6
    nll = 0.5 * torch.log(variance) + 0.5 * ((true_change - mu) ** 2) / variance
    price_loss = torch.mean(nll)

    # BCE for direction
    bce_loss = nn.BCELoss()(direction_prob, true_direction.float())

    total = price_loss + bce_loss
    return total, price_loss, bce_loss
