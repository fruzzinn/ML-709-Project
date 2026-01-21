# ML-709: Designing Robust Agentic AI Workflows under Adversarial Tool Use

Research project investigating robustness mechanisms for LLM-based agentic systems operating with potentially adversarial tools.

## Research Goals

1. **Implement a simple agentic workflow** - LLM + memory + tools using the ReAct pattern
2. **Simulate adversarial tools** - Wrong outputs, delayed responses, poisoned APIs
3. **Analyze failure propagation** - Track how failures cascade through reasoning, memory, and planning
4. **Propose workflow-level robustness mechanisms** - Tool verification, redundancy, rollback
5. **Evaluate effectiveness** - Task success rate, safety violations, latency overhead

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- GPU with CUDA support (for vLLM) or CPU-only mode available

### Installation

```bash
# Clone the repository
git clone https://github.com/fruzzinn/ML-709-Project.git
cd ML-709-Project

# Install dependencies with uv
uv sync

# Copy environment configuration
cp .env.example .env
```

### Running Experiments

#### Option 1: With Docker (Recommended)

```bash
# Start vLLM server and run experiment
docker-compose up experiment

# Run ADRS optimization loop
docker-compose --profile adrs up adrs
```

#### Option 2: Local Development

```bash
# Start vLLM server for a specific model (8-bit quantized)
./scripts/start_vllm.sh mistral-7b

# Or manually:
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 --port 8000 \
    --quantization bitsandbytes \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# Run baseline experiment
uv run python scripts/run_experiment.py --config configs/default.yaml

# Run with attack scenario
uv run python scripts/run_experiment.py --config configs/attacks/wrong_output.yaml

# Run ADRS optimization loop (uses Claude Code Bridge - no API key needed!)
uv run python scripts/run_adrs_loop.py --generations 10

# Run sequential experiments across ALL models
uv run python scripts/run_sequential_models.py --config configs/models.yaml
```

### Models Under Test (8-bit Quantized)

All models are tested with 8-bit quantization via bitsandbytes for efficient inference:

| Model | Parameters | HuggingFace ID |
|-------|------------|----------------|
| GLM-4-9B | 9B | `THUDM/glm-4-9b-chat` |
| Llama 3.1 8B | 8B | `meta-llama/Llama-3.1-8B-Instruct` |
| Qwen2.5-VL-7B | 7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Mistral 7B | 7B | `mistralai/Mistral-7B-Instruct-v0.3` |
| Gemma 2B | 2B | `google/gemma-2b-it` |
| Gemma 7B | 7B | `google/gemma-7b-it` |
| TinyLlama | 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Phi-3 Mini | 3.8B | `microsoft/Phi-3-mini-4k-instruct` |

### Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
ML-709-Project/
├── configs/                 # Experiment configurations
│   ├── default.yaml         # Baseline (no attacks)
│   ├── attacks/             # Attack scenario configs
│   └── defenses/            # Defense mechanism configs
├── src/
│   ├── agent/               # ReAct orchestrator
│   │   ├── orchestrator.py  # Main agent loop
│   │   ├── state.py         # State management
│   │   └── memory.py        # Working memory
│   ├── tools/               # Tool system
│   │   ├── registry.py      # Tool registry
│   │   ├── honest/          # Honest tool implementations
│   │   └── adversarial/     # Attack wrappers
│   ├── attacks/             # Attack simulation
│   │   ├── attack_types.py  # Attack definitions
│   │   ├── scheduler.py     # Attack scheduling
│   │   └── benchmarks/      # BAD-ACTS, TAMAS, AgentHarm
│   ├── defenses/            # Defense mechanisms
│   │   ├── tool_verification.py
│   │   ├── redundancy.py
│   │   ├── rollback.py
│   │   └── consistency_checker.py
│   ├── evaluation/          # Metrics and analysis
│   │   ├── metrics.py
│   │   ├── failure_propagation.py
│   │   └── reporter.py
│   ├── adrs/                # AI-Driven Research System
│   │   ├── inner_loop/      # MAP-Elites selection
│   │   └── outer_loop/      # BFTS experiment manager
│   └── llm/                 # LLM abstraction
├── scripts/                 # Execution scripts
├── tests/                   # Test suite
└── experiments/             # Output directory
```

## Attack Types

| Attack | Description |
|--------|-------------|
| **Wrong Output** | Returns plausible but incorrect results |
| **Delayed Response** | Introduces latency and timeouts |
| **Poisoned API** | Injects malicious content into responses |
| **Byzantine** | Inconsistent/unpredictable behavior |
| **Collusion** | Coordinated multi-tool attacks |

## Defense Mechanisms

| Defense | Description |
|---------|-------------|
| **Tool Verification** | Type checking, range validation, injection detection |
| **Redundancy** | Multi-source consensus verification |
| **Rollback** | State checkpointing and recovery |
| **Self-Consistency** | Multi-check reasoning validation |

## Evaluation Metrics

- **Task Success Rate** - Percentage of tasks completed correctly
- **Safety Score** - Violations blocked / violations attempted
- **Robustness Score** - Attacks detected / attacks received
- **Latency Overhead** - Additional time from defense mechanisms
- **Failure Cascade Depth** - How far failures propagate

## Benchmarks

The project integrates with three adversarial agent benchmarks:

| Benchmark | Instances | Description |
|-----------|-----------|-------------|
| [BAD-ACTS](https://arxiv.org/abs/2508.16481) | 188 | Harmful action detection |
| [TAMAS](https://arxiv.org/abs/2511.05269) | 300 | Multi-agent security |
| [AgentHarm](https://openreview.net/forum?id=AC5n7xHuR1) | 110 | Safety evaluation |

## Configuration

Experiments are configured via YAML files. See `configs/default.yaml` for the full schema.

```yaml
experiment:
  name: "my_experiment"
  output_dir: "experiments/my_experiment"

agent:
  max_loops: 10
  temperature: 0.7

attacks:
  enabled: true
  type: "wrong_output"
  probability: 0.3

defenses:
  tool_verification:
    enabled: true
  self_consistency:
    enabled: true
    threshold: 0.7
```

## ADRS: AI-Driven Research System

The project includes an automated experimentation system based on:

- **Inner Loop**: MAP-Elites quality-diversity selection (inspired by OpenEvolve)
- **Outer Loop**: Best-First Tree Search for experiment exploration (inspired by AI-Scientist-v2)

This enables automated discovery and evaluation of novel defense mechanisms.

### Architecture Separation

| Component | Model | Purpose |
|-----------|-------|---------|
| **Agent Under Test** | Mistral (vLLM) | Subject of adversarial research |
| **ADRS Research System** | Claude Opus 4.5 | Defense design, failure analysis |

### Claude Code Bridge (No API Key Needed!)

ADRS uses the **Claude Code Bridge** - it leverages your Claude CLI authentication:

```bash
# Just make sure you're logged into Claude CLI
claude  # Login if needed

# Run ADRS - uses your Claude subscription, no API key required!
uv run python scripts/run_adrs_loop.py --generations 10
```

The bridge calls the `claude` CLI under the hood, using your existing OAuth authentication.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this work, please cite:

```bibtex
@misc{ml709-adversarial-agents,
  title={Designing Robust Agentic AI Workflows under Adversarial Tool Use},
  author={ML-709 Research Team},
  year={2025},
  url={https://github.com/fruzzinn/ML-709-Project}
}
```
