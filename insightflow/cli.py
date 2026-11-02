"""Analyze a file: python -m insightflow.cli examples/discussion.txt --mode demo."""
import argparse
import json
from pathlib import Path
from .pipeline import analyze, plain_text
from .schemas import AnalyzeRequest, Options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--mode', choices=['ai','demo'], default='ai')
    parser.add_argument('--topics', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('result.json'))
    parser.add_argument('--format', choices=['json','txt'], default='json')
    args = parser.parse_args()
    try:
        text = args.input.read_text(encoding='utf-8-sig')
        data = {'discussion': json.loads(text)} if args.input.suffix.lower() == '.json' else {'text': text}
        result = analyze(AnalyzeRequest(**data, options=Options(mode=args.mode, num_topics=args.topics)), lambda s: print(s, flush=True))
        output = json.dumps(result, indent=2, ensure_ascii=False) if args.format == 'json' else plain_text(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
        print('Saved:', args.output.resolve())
    except Exception as exc:
        parser.exit(1, f'Analysis failed: {exc}\n')

if __name__ == '__main__':
    main()
