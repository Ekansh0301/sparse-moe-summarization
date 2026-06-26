import torch
import numpy as np
import string
import json
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Dict, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'for',
    'if', 'in', 'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such',
    'that', 'the', 'their', 'then', 'there', 'these', 'they', 'this', 'to',
    'was', 'will', 'with', 'he', 'she', 'we', 'you', 'i', 'has', 'have',
    'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might',
    'can', 'said', 'says', 'am', 'were', 'being', 'been', 'from', 'up',
    'about', 'after', 'all', 'also', 'when', 'where', 'which', 'who', 'whom'
}


def load_xsum_test(max_samples: Optional[int] = None):
    dataset = load_dataset("EdinburghNLP/xsum", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def compute_rouge_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    
    bleu_score = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references]
    )
    
    return {
        'rouge1': round(rouge_scores['rouge1'] * 100, 2),
        'rouge2': round(rouge_scores['rouge2'] * 100, 2),
        'rougeL': round(rouge_scores['rougeL'] * 100, 2),
        'bleu': round(bleu_score['bleu'] * 100, 2)
    }


def compute_bertscore(predictions: List[str], references: List[str], 
                     device: str = 'cpu', batch_size: int = 16) -> Dict[str, float]:
    bertscore = evaluate.load("bertscore")
    
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type="distilbert-base-uncased",
        device=device,
        batch_size=batch_size,
        lang="en"
    )
    
    return {
        'bertscore_precision': round(np.mean(results['precision']) * 100, 2),
        'bertscore_recall': round(np.mean(results['recall']) * 100, 2),
        'bertscore_f1': round(np.mean(results['f1']) * 100, 2)
    }


def compute_compression_ratio(predictions: List[str], documents: List[str]) -> Dict[str, float]:
    ratios = []
    for pred, doc in zip(predictions, documents):
        pred_len = len(pred.split())
        doc_len = len(doc.split())
        if doc_len > 0:
            ratios.append(pred_len / doc_len)
    
    return {
        'compression_ratio': round(np.mean(ratios) * 100, 2),
        'compression_std': round(np.std(ratios) * 100, 2)
    }


def compute_extractiveness(predictions: List[str], documents: List[str]) -> Dict[str, float]:
    extractiveness_scores = []
    
    for pred, doc in zip(predictions, documents):
        doc_words = set([
            w.lower().strip(string.punctuation)
            for w in doc.split()
            if (w.lower().strip(string.punctuation) not in STOPWORDS and
                len(w.strip(string.punctuation)) > 2 and
                w.strip(string.punctuation).isalpha())
        ])
        
        pred_words = [
            w.lower().strip(string.punctuation)
            for w in pred.split()
            if (w.lower().strip(string.punctuation) not in STOPWORDS and
                len(w.strip(string.punctuation)) > 2 and
                w.strip(string.punctuation).isalpha())
        ]
        
        if len(pred_words) > 0:
            overlap = sum(1 for w in pred_words if w in doc_words)
            extractiveness_scores.append((overlap / len(pred_words)) * 100)
    
    return {
        'extractiveness': round(np.mean(extractiveness_scores), 2),
        'extractiveness_std': round(np.std(extractiveness_scores), 2)
    }


def compute_factual_consistency(predictions: List[str], documents: List[str],
                                use_summac: bool = False, device: str = 'cpu') -> Dict[str, float]:
    if use_summac:
        try:
            from summac.model_summac import SummaCZS
            model = SummaCZS(granularity="sentence", model_name="vitc", device=device)
            
            scores = []
            eval_samples = min(100, len(predictions))
            for i in tqdm(range(eval_samples), desc="Computing SummaC"):
                score = model.score([documents[i]], [predictions[i]])
                scores.append(score['scores'][0] * 100)
            
            return {
                'factual_consistency': round(np.mean(scores), 2),
                'factual_std': round(np.std(scores), 2),
                'method': 'summac'
            }
        except ImportError:
            print("SummaC not installed, using entity overlap fallback")
            use_summac = False
    
    scores = []
    for pred, doc in zip(predictions, documents):
        doc_entities = set([
            w for w in doc.split()
            if len(w) > 0 and (w[0].isupper() or w.isdigit())
        ])
        
        pred_entities = [
            w for w in pred.split()
            if len(w) > 0 and (w[0].isupper() or w.isdigit())
        ]
        
        if len(pred_entities) > 0:
            overlap = sum(1 for e in pred_entities if e in doc_entities)
            scores.append((overlap / len(pred_entities)) * 100)
        else:
            scores.append(100.0)
    
    return {
        'factual_consistency': round(np.mean(scores), 2),
        'factual_std': round(np.std(scores), 2),
        'method': 'entity_overlap'
    }


def evaluate_summaries(predictions: List[str], 
                      references: List[str], 
                      documents: List[str],
                      use_summac: bool = False,
                      device: str = 'cpu',
                      batch_size: int = 16,
                      verbose: bool = True) -> Dict[str, float]:
    
    if len(predictions) != len(references) or len(predictions) != len(documents):
        raise ValueError("predictions, references, and documents must have same length")
    
    results = {}
    
    if verbose:
        print(f"\nEvaluating {len(predictions)} summaries...")
        print("="*60)
    
    if verbose:
        print("\n[1/5] Computing ROUGE & BLEU...")
    results.update(compute_rouge_bleu(predictions, references))
    
    if verbose:
        print("[2/5] Computing BERTScore...")
    results.update(compute_bertscore(predictions, references, device, batch_size))
    
    if verbose:
        print("[3/5] Computing Compression Ratio...")
    results.update(compute_compression_ratio(predictions, documents))
    
    if verbose:
        print("[4/5] Computing Extractiveness...")
    results.update(compute_extractiveness(predictions, documents))
    
    if verbose:
        print("[5/5] Computing Factual Consistency...")
    results.update(compute_factual_consistency(predictions, documents, use_summac, device))
    
    avg_pred_len = np.mean([len(p.split()) for p in predictions])
    avg_ref_len = np.mean([len(r.split()) for r in references])
    avg_doc_len = np.mean([len(d.split()) for d in documents])
    
    results['avg_pred_length'] = round(avg_pred_len, 2)
    results['avg_ref_length'] = round(avg_ref_len, 2)
    results['avg_doc_length'] = round(avg_doc_len, 2)
    results['num_samples'] = len(predictions)
    
    if verbose:
        print("\n" + "="*60)
        print("RESULTS:")
        print("-"*60)
        print(f"ROUGE-1:              {results['rouge1']:.2f}")
        print(f"ROUGE-2:              {results['rouge2']:.2f}")
        print(f"ROUGE-L:              {results['rougeL']:.2f}")
        print(f"BLEU:                 {results['bleu']:.2f}")
        print(f"BERTScore F1:         {results['bertscore_f1']:.2f}")
        print(f"Compression Ratio:    {results['compression_ratio']:.2f}%")
        print(f"Extractiveness:       {results['extractiveness']:.2f}%")
        print(f"Factual Consistency:  {results['factual_consistency']:.2f}%")
        print("="*60 + "\n")
    
    return results


def evaluate_model_on_xsum(model, tokenizer, max_samples: Optional[int] = None,
                           batch_size: int = 8, device: str = 'cuda',
                           generation_config: Optional[Dict] = None,
                           use_summac: bool = False,
                           save_path: Optional[str] = None) -> Dict[str, float]:
    
    test_data = load_xsum_test(max_samples)
    
    if generation_config is None:
        generation_config = {
            'max_length': 64,
            'num_beams': 4,
            'length_penalty': 2.0,
            'early_stopping': True,
            'no_repeat_ngram_size': 3
        }
    
    model.eval()
    model.to(device)
    predictions = []
    
    print(f"Generating summaries for {len(test_data)} examples...")
    for i in tqdm(range(0, len(test_data), batch_size), desc="Inference"):
        batch = test_data[i:i+batch_size]
        
        inputs = tokenizer(
            batch["document"],
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
    
    references = test_data["summary"]
    documents = test_data["document"]
    
    results = evaluate_summaries(
        predictions=predictions,
        references=references,
        documents=documents,
        use_summac=use_summac,
        device=device,
        batch_size=batch_size
    )
    
    if save_path:
        output_data = {
            'results': results,
            'predictions': predictions[:10],
            'references': references[:10]
        }
        with open(save_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {save_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate summarization models')
    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--max-samples', type=int, help='Max samples to evaluate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--use-summac', action='store_true', help='Use SummaC for factual consistency')
    parser.add_argument('--save', type=str, help='Path to save results')
    
    args = parser.parse_args()
    
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    
    results = evaluate_model_on_xsum(
        model=model,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        use_summac=args.use_summac,
        save_path=args.save
    )
