"""Run once after installing requirements-ai.txt. Models are reused from cache."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insightflow.models import get_models
if __name__ == '__main__':
    print('Downloading/loading MPNet and DistilBART. This can take several minutes.')
    model = get_models('ai')
    print('Models ready:', model.name)
    print('Embedding shape:', model.embed(['Reliable public transport reduces commuting time.']).shape)
    print('Generated summary:', model.summarize('The city plans to improve public transport. Frequent buses will reduce waiting times for commuters. Protected bicycle lanes will make cycling safer.', 60))
