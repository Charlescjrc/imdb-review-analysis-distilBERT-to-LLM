# Contributing to IMDb Sentiment Analysis Pipeline 🎬

Thank you for your interest in contributing! This guide will help you get started.

## 📜 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism gracefully
- Focus on what's best for the community

## 🚀 Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/imdb-review-analysis-distilBERT-to-LLM.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Push and create a Pull Request

## 💡 How to Contribute

### Report Bugs 🐛
Create an issue with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)

### Suggest Features ✨
Open an issue describing:
- Use case and motivation
- Proposed solution
- Alternative approaches considered

### Submit Code 🔧

**Priority Areas:**
- Model improvements & benchmarking
- Multi-language support
- Performance optimizations
- Documentation & tutorials
- Test coverage

## 🛠️ Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Configure Hugging Face
huggingface-cli login
```

## 📤 Pull Request Process

1. **Before submitting:**
   - Run tests: `pytest`
   - Check formatting: `black . && flake8 .`
   - Update documentation

2. **PR Guidelines:**
   - Use descriptive titles
   - Reference related issues
   - Include benchmark results if performance-related
   - Ensure all CI checks pass

## 🎨 Style Guide

- **Code:** We use [Black](https://github.com/psf/black) formatter and follow PEP 8
- **Commits:** Follow [Conventional Commits](https://www.conventionalcommits.org/)
  - `feat:` New feature
  - `fix:` Bug fix
  - `docs:` Documentation
  - `perf:` Performance improvement
  - `test:` Testing
  - `refactor:` Code refactoring


## 🤝 Getting Help

- 💬 [Discussions](https://github.com/ifieryarrows/imdb-review-analysis-distilBERT-to-LLM/discussions) - Questions & ideas
- 🐛 [Issues](https://github.com/ifieryarrows/imdb-review-analysis-distilBERT-to-LLM/issues) - Bugs & features
- 🏷️ Look for `good-first-issue` labels if you're new

---

Contributors will be credited in [CONTRIBUTORS.md](CONTRIBUTORS.md) and release notes.

Thank you for helping make film sentiment analysis more powerful! 🎬✨
