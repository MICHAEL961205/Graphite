# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import pandas as pd
import numpy as np
import os

ROOT_DIR = "tests"
SAVE_DIR = "evaluation_results"

def summarize_algorithm_results(algorithm_name):
    """Summarize results for a specific algorithm."""
    scores_file = os.path.join(ROOT_DIR, SAVE_DIR, f"{algorithm_name}_scores.csv")
    times_file = os.path.join(ROOT_DIR, SAVE_DIR, f"{algorithm_name}_run_times.csv")
    
    if not os.path.exists(scores_file) or not os.path.exists(times_file):
        return None
    
    scores_df = pd.read_csv(scores_file, index_col=0)
    times_df = pd.read_csv(times_file, index_col=0)
    
    # Get algorithm and NN scores
    if algorithm_name in scores_df.columns:
        algo_scores = scores_df[algorithm_name].values
    else:
        return None
        
    if 'NearestNeighbourSolver' in scores_df.columns:
        nn_scores = scores_df['NearestNeighbourSolver'].values
    else:
        return None
    
    # Get algorithm and NN times
    if algorithm_name in times_df.columns:
        algo_times = times_df[algorithm_name].values
    else:
        return None
        
    if 'NearestNeighbourSolver' in times_df.columns:
        nn_times = times_df['NearestNeighbourSolver'].values
    else:
        return None
    
    # Filter out infinite values
    valid_algo_scores = [s for s in algo_scores if s != float('inf')]
    valid_nn_scores = [s for s in nn_scores if s != float('inf')]
    valid_algo_times = [t for t in algo_times if t != float('inf')]
    valid_nn_times = [t for t in nn_times if t != float('inf')]
    
    if not valid_algo_scores or not valid_nn_scores:
        return None
    
    # Calculate improvements
    improvements = sum(1 for a, n in zip(algo_scores, nn_scores) 
                      if a != float('inf') and n != float('inf') and a < n)
    total_comparable = sum(1 for a, n in zip(algo_scores, nn_scores) 
                          if a != float('inf') and n != float('inf'))
    
    return {
        'algorithm': algorithm_name,
        'problems_solved': len(valid_algo_scores),
        'total_problems': len(algo_scores),
        'improvements': improvements,
        'total_comparable': total_comparable,
        'improvement_rate': improvements / total_comparable if total_comparable > 0 else 0,
        'avg_score': np.mean(valid_algo_scores),
        'avg_nn_score': np.mean(valid_nn_scores),
        'score_improvement': (np.mean(valid_nn_scores) - np.mean(valid_algo_scores)) / np.mean(valid_nn_scores) * 100,
        'avg_time': np.mean(valid_algo_times),
        'avg_nn_time': np.mean(valid_nn_times),
        'time_ratio': np.mean(valid_algo_times) / np.mean(valid_nn_times) if np.mean(valid_nn_times) > 0 else float('inf')
    }

def main():
    algorithms = [
        'TwoOptSolver',
        'ThreeOptSolver', 
        'SimulatedAnnealingSolver',
        'GeneticAlgorithmSolver'
    ]
    
    print("=" * 80)
    print("TSP ALGORITHM EVALUATION SUMMARY")
    print("=" * 80)
    print()
    
    results = []
    
    for algo in algorithms:
        result = summarize_algorithm_results(algo)
        if result:
            results.append(result)
            
            print(f"Algorithm: {result['algorithm']}")
            print(f"  Problems solved: {result['problems_solved']}/{result['total_problems']}")
            print(f"  Improvements over NN: {result['improvements']}/{result['total_comparable']} ({result['improvement_rate']:.1%})")
            print(f"  Average score: {result['avg_score']:.2f}")
            print(f"  NN average score: {result['avg_nn_score']:.2f}")
            print(f"  Score improvement: {result['score_improvement']:.2f}%")
            print(f"  Average time: {result['avg_time']:.2f}s")
            print(f"  NN average time: {result['avg_nn_time']:.2f}s")
            print(f"  Time ratio: {result['time_ratio']:.2f}x")
            print()
    
    # Create summary DataFrame
    if results:
        summary_df = pd.DataFrame(results)
        summary_df = summary_df.sort_values('score_improvement', ascending=False)
        
        print("=" * 80)
        print("RANKING BY SCORE IMPROVEMENT")
        print("=" * 80)
        print()
        
        for i, (_, row) in enumerate(summary_df.iterrows(), 1):
            status = "✅ BETTER" if row['improvement_rate'] > 0 else "❌ WORSE"
            print(f"{i}. {row['algorithm']} - {row['score_improvement']:.2f}% improvement {status}")
        
        print()
        print("=" * 80)
        print("DETAILED COMPARISON TABLE")
        print("=" * 80)
        print()
        
        # Create a nice table
        table_data = []
        for _, row in summary_df.iterrows():
            table_data.append([
                row['algorithm'],
                f"{row['improvements']}/{row['total_comparable']}",
                f"{row['improvement_rate']:.1%}",
                f"{row['score_improvement']:.2f}%",
                f"{row['avg_time']:.1f}s",
                f"{row['time_ratio']:.1f}x"
            ])
        
        df_table = pd.DataFrame(table_data, columns=[
            'Algorithm', 'Improvements', 'Rate', 'Score %', 'Avg Time', 'Time Ratio'
        ])
        print(df_table.to_string(index=False))
        
        # Save summary
        summary_df.to_csv(os.path.join(ROOT_DIR, SAVE_DIR, "algorithm_summary.csv"))
        print(f"\nSummary saved to: {os.path.join(ROOT_DIR, SAVE_DIR, 'algorithm_summary.csv')}")
    
    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if results:
        best_algorithm = summary_df.iloc[0]
        if best_algorithm['improvement_rate'] > 0:
            print(f"✅ Best performing algorithm: {best_algorithm['algorithm']}")
            print(f"   - {best_algorithm['score_improvement']:.2f}% better than NearestNeighbourSolver")
            print(f"   - {best_algorithm['improvements']}/{best_algorithm['total_comparable']} problems improved")
        else:
            print("❌ No algorithm significantly outperformed NearestNeighbourSolver")
            print("   - All algorithms either performed worse or had similar performance")
    else:
        print("❌ No valid results found")

if __name__ == "__main__":
    main()
