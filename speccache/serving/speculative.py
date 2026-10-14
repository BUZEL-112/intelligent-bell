"""Target-model parallel speculative verification engine for near-miss semantic cache hits.

When a prompt achieves a semantic similarity score s in [0.80, 0.92), SpecCache uses
cached continuation tokens Y_cached = [y_1, ..., y_gamma] (default gamma = 6).
Instead of gamma sequential autoregressive forward passes, SpecCache executes a single
parallel target-model verification forward pass, accepts the longest contiguous valid
prefix, and emits 1 bonus token from the target logits at the first mismatch index.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch

from speccache.serving.kv_injector import SpecDynamicCache


@dataclass
class SpeculativeDraft:
    """Candidate draft continuation tokens retrieved from a semantically similar cached entry."""
    draft_tokens: List[int]
    similarity_score: float
    matched_chunk_idx: int = 0
    cached_prompt_prefix: Optional[List[int]] = None


@dataclass
class VerificationResult:
    """Detailed result of target-model speculative verification."""
    accepted_tokens: List[int]
    bonus_token: int
    num_accepted: int
    total_emitted: int
    acceptance_rate: float
    rejection_index: Optional[int] = None
    target_tokens: List[int] = field(default_factory=list)

    @property
    def emitted_tokens(self) -> List[int]:
        """All tokens emitted by this verification step (accepted tokens + bonus token)."""
        return self.accepted_tokens + [self.bonus_token]


class SpeculativeVerifier:
    """Executes target-model parallel speculative verification and KV cache rollbacks.

    Features:
    - Parallel verification: Verifies gamma tokens in a single target model forward pass.
    - Speculative Acceptance: Accepts longest contiguous prefix where argmax P_target == y_i.
    - Bonus Token: Emits 1 bonus token directly from target model logits at the mismatch position.
    - KV Cache Rollback: Truncates key/value states for rejected speculative positions.
    """

    def __init__(
        self,
        gamma: int = 6,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        """Initialize SpeculativeVerifier.

        Args:
            gamma: Candidate draft continuation length (default 6).
            temperature: Sampling temperature (0.0 for deterministic greedy verification).
            top_p: Nucleus sampling probability parameter.
        """
        if gamma <= 0:
            raise ValueError(f"gamma draft length must be positive, got {gamma}")
        self.gamma = gamma
        self.temperature = temperature
        self.top_p = top_p

    def verify_draft_tokens(
        self,
        draft_tokens: List[int],
        verification_logits: torch.Tensor,
        greedy: bool = True,
    ) -> VerificationResult:
        """Verify candidate draft tokens against target model verification logits.

        Verification Logits Layout:
        Tensor of shape [num_draft + 1, vocab_size] or [1, num_draft + 1, vocab_size].
        - verification_logits[0]: Predicts the first draft token y_1 (index 0).
        - verification_logits[1]: Predicts the second draft token y_2 (index 1).
        ...
        - verification_logits[gamma - 1]: Predicts draft token y_gamma (index gamma - 1).
        - verification_logits[gamma]: Predicts bonus token y_{gamma + 1} if all draft tokens pass!

        Args:
            draft_tokens: List of candidate draft token IDs [y_1, ..., y_gamma].
            verification_logits: Logits output by the target model parallel forward pass.
            greedy: Whether to use greedy argmax matching or stochastic sampling.

        Returns:
            VerificationResult containing accepted tokens, bonus token, and stats.
        """
        logits = verification_logits
        if logits.ndim == 3:
            logits = logits.squeeze(0)  # [num_draft + 1, vocab_size]

        num_draft = len(draft_tokens)
        if logits.shape[0] < num_draft + 1:
            raise ValueError(
                f"verification_logits length {logits.shape[0]} is insufficient for "
                f"{num_draft} draft tokens (requires at least {num_draft + 1})."
            )

        if greedy or self.temperature == 0.0:
            return self._verify_greedy(draft_tokens, logits)
        else:
            return self._verify_stochastic(draft_tokens, logits)

    def _verify_greedy(
        self,
        draft_tokens: List[int],
        logits: torch.Tensor,
    ) -> VerificationResult:
        """Greedy argmax verification matching."""
        num_draft = len(draft_tokens)
        accepted: List[int] = []
        target_tokens: List[int] = []
        rejection_idx: Optional[int] = None
        bonus_token: int = -1

        # Check each candidate draft token against the target model prediction
        for j in range(num_draft):
            pred_token = int(logits[j].argmax(dim=-1).item())
            target_tokens.append(pred_token)

            if pred_token == draft_tokens[j]:
                accepted.append(draft_tokens[j])
            else:
                # First mismatch: draft rejected here!
                rejection_idx = j
                # Bonus token is the target model's actual prediction at this position
                bonus_token = pred_token
                break
        else:
            # All candidate draft tokens were accepted!
            # Bonus token comes from the extra position (logits[num_draft])
            bonus_token = int(logits[num_draft].argmax(dim=-1).item())
            target_tokens.append(bonus_token)

        num_accepted = len(accepted)
        return VerificationResult(
            accepted_tokens=accepted,
            bonus_token=bonus_token,
            num_accepted=num_accepted,
            total_emitted=num_accepted + 1,
            acceptance_rate=num_accepted / num_draft if num_draft > 0 else 0.0,
            rejection_index=rejection_idx,
            target_tokens=target_tokens,
        )

    def _verify_stochastic(
        self,
        draft_tokens: List[int],
        logits: torch.Tensor,
    ) -> VerificationResult:
        """Stochastic verification with speculative rejection sampling."""
        num_draft = len(draft_tokens)
        accepted: List[int] = []
        target_tokens: List[int] = []
        rejection_idx: Optional[int] = None
        bonus_token: int = -1

        probs = torch.softmax(logits / max(self.temperature, 1e-4), dim=-1)

        for j in range(num_draft):
            candidate = draft_tokens[j]
            target_p = float(probs[j, candidate].item())

            # For greedy approximation or uniform rejection sampling:
            # In standard speculative decoding with uniform/known draft prior, accept with min(1, p_target / p_draft)
            # When draft prior is uniform or unknown, threshold against random draw:
            r = torch.rand(1).item()
            if r <= target_p or target_p >= 0.5:
                accepted.append(candidate)
                target_tokens.append(candidate)
            else:
                rejection_idx = j
                # Sample bonus token from target model distribution
                bonus_token = int(torch.multinomial(probs[j], num_samples=1).item())
                target_tokens.append(bonus_token)
                break
        else:
            # All accepted
            bonus_token = int(torch.multinomial(probs[num_draft], num_samples=1).item())
            target_tokens.append(bonus_token)

        num_accepted = len(accepted)
        return VerificationResult(
            accepted_tokens=accepted,
            bonus_token=bonus_token,
            num_accepted=num_accepted,
            total_emitted=num_accepted + 1,
            acceptance_rate=num_accepted / num_draft if num_draft > 0 else 0.0,
            rejection_index=rejection_idx,
            target_tokens=target_tokens,
        )

    def rollback_cache(
        self,
        cache: SpecDynamicCache,
        valid_seq_len: int,
    ) -> None:
        """Roll back KV cache to discard key/value states beyond valid_seq_len.

        Args:
            cache: SpecDynamicCache instance.
            valid_seq_len: Prefix length containing accepted tokens + bonus token.
        """
        cache.truncate(valid_seq_len)
