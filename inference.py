"""
inference.py — RadiotherapyPlanningEnv-v1
==========================================
Uses an LLM (via OpenAI-compatible API) to plan radiotherapy treatment
across all 3 tasks: prostate (easy), head & neck (medium), pediatric brain (hard).

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (strictly followed):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import textwrap
from typing import List, Optional

import gymnasium as gym
from openai import OpenAI

import radiotherapy_env  # noqa: F401 — registers gym envs

# ── Configuration ─────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "radiotherapy-planning"
SEED = 42
SUCCESS_SCORE_THRESHOLD = 0.6

TASKS = [
    ("prostate_easy",        "RadiotherapyEnv-prostate-v1"),
    ("head_neck_medium",     "RadiotherapyEnv-headneck-v1"),
    ("pediatric_brain_hard", "RadiotherapyEnv-pediatricbrain-v1"),
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert AI radiation oncologist planning cancer radiotherapy treatment.
    Your goal: place and optimize radiation beams to maximize tumor dose while
    protecting nearby organs-at-risk (OARs).

    Available actions (reply with ONLY the integer):
      0 - Add beam at next default angle
      1 - Rotate last beam +10 degrees
      2 - Rotate last beam -10 degrees
      3 - Increase last beam dose weight by 10%
      4 - Decrease last beam dose weight by 10%
      5 - Remove last beam
      6 - Fine-tune all beams (small random perturbation)
      7 - Lock plan and terminate episode

    Strategy:
      - Steps 1-7:  Add beams (action 0) until you have 6-7 active beams.
      - Steps 8-35: Rotate and fine-tune (actions 1, 2, 3, 4, 6).
      - Steps 36+:  If tumor_uncoverage < 0.15 and max OAR violation < 0.3,
                    lock the plan (action 7). Otherwise keep fine-tuning.
      - Never lock early — always optimize first.

    Reply with ONLY a single digit 0-7. No explanation, no punctuation.
""").strip()


# ── Structured logging ────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── Observation → text prompt ─────────────────────────────────────────────────

def format_observation(obs: dict, step: int, max_steps: int) -> str:
    n_beams = int(obs["beams"][:, 2].sum())
    constraints = obs["constraints"]
    tumor_uncov = float(constraints[0])
    oar_violations = [f"{float(v):.2f}" for v in constraints[1:]]
    # DVH index 47 ≈ V95 (fraction of tumor receiving ≥95% prescription dose)
    tumor_v95 = float(obs["dvh_tumor"][47]) if len(obs["dvh_tumor"]) > 47 else 0.0

    return (
        f"Step {step}/{max_steps} | Active beams: {n_beams}/7\n"
        f"Tumor V95 (coverage): {tumor_v95:.2f} (target ≥0.95)\n"
        f"Tumor uncoverage:     {tumor_uncov:.2f} (lower is better, target <0.15)\n"
        f"OAR violations:       {oar_violations} (lower is better, target <0.30)"
    )


# ── LLM action selection ──────────────────────────────────────────────────────

def get_llm_action(client: OpenAI, obs: dict, step: int, max_steps: int) -> int:
    obs_text = format_observation(obs, step, max_steps)
    user_prompt = f"Current treatment plan state:\n{obs_text}\n\nChoose action (0-7):"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=5,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        for token in text.split():
            try:
                action = int(token)
                if 0 <= action <= 7:
                    return action
            except ValueError:
                continue
    except Exception as exc:
        print(f"[DEBUG] LLM request failed at step {step}: {exc}", flush=True)

    return _heuristic_fallback(obs, step, max_steps)


def _heuristic_fallback(obs: dict, step: int, max_steps: int) -> int:
    """Fallback when LLM call fails or returns unparseable output."""
    n_beams = int(obs["beams"][:, 2].sum())
    constraints = obs["constraints"]
    tumor_uncov = float(constraints[0])
    oar_max = float(constraints[1:].max()) if len(constraints) > 1 else 0.0
    frac = step / max_steps

    if n_beams < 6 and frac < 0.40:
        return 0  # add beam
    elif frac > 0.85 and tumor_uncov < 0.20 and oar_max < 0.40:
        return 7  # lock plan
    elif tumor_uncov > 0.40:
        return 3  # increase dose
    else:
        return 6  # fine-tune


# ── Single episode runner ─────────────────────────────────────────────────────

def run_episode(client: OpenAI, task_id: str, env_id: str) -> float:
    env = gym.make(env_id)
    max_steps = env.spec.max_episode_steps or 50

    obs, _ = env.reset(seed=SEED)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, max_steps + 1):
            action = get_llm_action(client, obs, step, max_steps)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward_f = float(reward)
            rewards.append(reward_f)
            steps_taken = step

            log_step(step=step, action=str(action), reward=reward_f, done=done, error=None)

            if done:
                break

        score = float(info.get("score", 0.0))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return score


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_id, env_id in TASKS:
        run_episode(client, task_id, env_id)


if __name__ == "__main__":
    main()
