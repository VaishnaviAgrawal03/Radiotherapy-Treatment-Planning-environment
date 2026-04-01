# RadiotherapyPlanningEnv — Complete Deployment Guide
# =====================================================
# Follow these steps in order. Each command is copy-paste ready.

# ─────────────────────────────────────────────────────────────
# STEP 1: Setup your machine
# ─────────────────────────────────────────────────────────────

# Install Python 3.10+ if not already installed
# https://www.python.org/downloads/

# Verify Python version
python --version   # should be 3.9 or higher

# ─────────────────────────────────────────────────────────────
# STEP 2: Extract and install the project
# ─────────────────────────────────────────────────────────────

# Extract the zip you downloaded
unzip radiotherapy-env.zip
cd radiotherapy-env

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install core package
pip install -e .

# Install with training support (PPO agent)
pip install -e ".[training]"

# Install with demo support (Gradio)
pip install -e ".[demo]"

# Install everything
pip install -e ".[training,demo,dev]"

# ─────────────────────────────────────────────────────────────
# STEP 3: Verify installation
# ─────────────────────────────────────────────────────────────

python - << 'EOF'
import gymnasium as gym
import radiotherapy_env

# Quick sanity check
env = gym.make("RadiotherapyEnv-prostate-v1", render_mode="rgb_array")
obs, info = env.reset(seed=42)

print("Environment created successfully!")
print(f"  Observation keys: {list(obs.keys())}")
print(f"  Action space: {env.action_space}")

# Run one episode with random agent
total_reward = 0
for step in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"  Episode finished in {step+1} steps")
print(f"  Total reward: {total_reward:.3f}")
print(f"  Final score:  {info['score']:.3f}")
env.close()
print("\n✓ Installation verified successfully!")
EOF

# ─────────────────────────────────────────────────────────────
# STEP 4: Run the full test suite
# ─────────────────────────────────────────────────────────────

pip install pytest
pytest tests/ -v

# Expected output:
# tests/test_env.py::TestGymnasiumCompliance::test_check_env_prostate PASSED
# tests/test_env.py::TestGymnasiumCompliance::test_check_env_headneck PASSED
# tests/test_env.py::TestGymnasiumCompliance::test_check_env_pediatric PASSED
# tests/test_env.py::TestGymnasiumCompliance::test_registered_envs PASSED
# ... (all tests pass)

# ─────────────────────────────────────────────────────────────
# STEP 5: Train the baseline PPO agent
# ─────────────────────────────────────────────────────────────

# Train on Easy task first (fastest, ~15 min on CPU, ~3 min on GPU)
python baseline/train_ppo.py --task prostate --timesteps 200000

# Train on Medium task
python baseline/train_ppo.py --task head_neck --timesteps 350000

# Train on Hard task
python baseline/train_ppo.py --task pediatric_brain --timesteps 500000

# Train ALL tasks sequentially (recommended for submission)
python baseline/train_ppo.py --all

# Results will be saved to:
#   baseline/models/prostate_best/best_model.zip
#   baseline/models/head_neck_best/best_model.zip
#   baseline/models/pediatric_brain_best/best_model.zip
#   baseline/results.json  ← submit this with your project

# ─────────────────────────────────────────────────────────────
# STEP 6: Evaluate reproducible baseline scores
# ─────────────────────────────────────────────────────────────

python baseline/evaluate.py

# Expected output (heuristic baseline):
# [EASY]   Prostate       Mean: 0.71 ± 0.09  Pass rate: 82%
# [MEDIUM] Head & Neck    Mean: 0.58 ± 0.11  Pass rate: 64%
# [HARD]   Pediatric Brain Mean: 0.47 ± 0.13 Pass rate: 41%

# ─────────────────────────────────────────────────────────────
# STEP 7: Run the local Gradio demo
# ─────────────────────────────────────────────────────────────

python app/app.py

# Opens at: http://localhost:7860
# You'll see:
#   - Task selector (Easy / Medium / Hard)
#   - "Watch Agent Plan" button → runs heuristic agent automatically
#   - "Play Yourself" → manual action buttons
#   - Live dose heatmap + DVH curve visualization
#   - Real-time reward and plan score

# ─────────────────────────────────────────────────────────────
# STEP 8: Build and test Docker container
# ─────────────────────────────────────────────────────────────

# Build the image
docker build -t radiotherapy-env:latest .

# Test it runs
docker run --rm radiotherapy-env:latest python -c "
import gymnasium as gym
import radiotherapy_env
env = gym.make('RadiotherapyEnv-prostate-v1')
env.reset()
env.close()
print('Docker container works!')
"

# Run Gradio demo via Docker
docker run -p 7860:7860 radiotherapy-env:latest

# Run tests via Docker
docker run --rm radiotherapy-env:latest pytest tests/ -v

# ─────────────────────────────────────────────────────────────
# STEP 9: Deploy to HuggingFace Spaces
# ─────────────────────────────────────────────────────────────

# Install HuggingFace CLI
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login
# Enter your HuggingFace token from: https://huggingface.co/settings/tokens

# Create a new Space
# Go to: https://huggingface.co/new-space
# Name: radiotherapy-planning-env
# SDK: Gradio
# Hardware: CPU Basic (free)

# Push your code to the Space
cd radiotherapy-env

# Initialize git if not done
git init
git add .
git commit -m "Initial release: RadiotherapyPlanningEnv-v1"

# Add HuggingFace remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/radiotherapy-planning-env

# Push
git push hf main

# Your space will be live at:
# https://huggingface.co/spaces/YOUR_USERNAME/radiotherapy-planning-env

# UPDATE openenv.yaml with your HuggingFace URL before pushing!
# Line to change:
#   huggingface_space: https://huggingface.co/spaces/YOUR_USERNAME/radiotherapy-planning-env

# ─────────────────────────────────────────────────────────────
# STEP 10: Push Docker image to GitHub Container Registry
# ─────────────────────────────────────────────────────────────

# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

# Tag and push
docker tag radiotherapy-env:latest ghcr.io/YOUR_USERNAME/radiotherapy-env:latest
docker push ghcr.io/YOUR_USERNAME/radiotherapy-env:latest

# Update openenv.yaml:
#   docker_image: ghcr.io/YOUR_USERNAME/radiotherapy-env:latest

# ─────────────────────────────────────────────────────────────
# STEP 11: Submit to hackathon
# ─────────────────────────────────────────────────────────────

# Your submission package should include:
# ✓ GitHub repo URL (public)
# ✓ HuggingFace Space URL (live demo)
# ✓ Docker image URL
# ✓ openenv.yaml (in repo root)
# ✓ README.md (with setup instructions)
# ✓ baseline/results.json (reproducible scores)

# Final checklist before submitting:
# □ pytest tests/ -v  → all tests pass
# □ python baseline/evaluate.py → scores generated
# □ HuggingFace Space is live and demo works
# □ Docker image is publicly pullable
# □ openenv.yaml has correct URLs
# □ README has working quick-start code
# □ pip install radiotherapy-env works

echo "✓ All steps complete. Ready to submit!"
