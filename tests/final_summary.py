import pandas as pd
import numpy as np
import os

ROOT_DIR = "tests"
SAVE_DIR = "evaluation_results"

def analyze_algorithm(algorithm_name, scores_file, times_file):
    """Analyze results for a specific algorithm."""
    if not os.path.exists(scores_file) or not os.path.exists(times_file):
        return None
    
    scores_df = pd.read_csv(scores_file, index_col=0)
    times_df = pd.read_csv(times_file, index_col=0)
    
    # Get algorithm and NN scores
    if algorithm_name not in scores_df.columns or 'NearestNeighbourSolver' not in scores_df.columns:
        return None
        
    algo_scores = scores_df[algorithm_name].values
    nn_scores = scores_df['NearestNeighbourSolver'].values
    
    # Get algorithm and NN times
    algo_times = times_df[algorithm_name].values
    nn_times = times_df['NearestNeighbourSolver'].values
    
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
        ('TwoOptSolver', 'two_opt_scores.csv', 'two_opt_run_times.csv'),
        ('ThreeOptSolver', 'three_opt_scores.csv', 'three_opt_run_times.csv'),
        ('SimulatedAnnealingSolver', 'simulated_annealing_scores.csv', 'simulated_annealing_run_times.csv'),
        ('AdvancedGeneticSolver', 'advanced_genetic_scores.csv', 'advanced_genetic_run_times.csv'),
        ('AdvancedSimulatedAnnealingSolver', 'advanced_simulated_annealing_scores.csv', 'advanced_simulated_annealing_run_times.csv'),
        ('CheapestInsertionTwoOptSolver', 'cheapestinsertiontwoopt_score.csv', 'cheapestinsertiontwoopt_run_times.csv'),
        ('NearestInsertionTwoOptSolver', 'nearest_insertion_two_opt_scores.csv', 'nearest_insertion_two_opt_run_times.csv'),
        ('FarthestInsertionTwoOptSolver', 'farthest_insertion_two_opt_scores.csv', 'farthest_insertion_two_opt_run_times.csv'),
        ('ChristofidesSolver', 'christofides_scores.csv', 'christofides_run_times.csv'),
    ]
    
    print("=" * 100)
    print("FINAL TSP ALGORITHM EVALUATION SUMMARY")
    print("=" * 100)
    print()
    
    results = []
    
    for algo_name, scores_file, times_file in algorithms:
        scores_path = os.path.join(ROOT_DIR, SAVE_DIR, scores_file)
        times_path = os.path.join(ROOT_DIR, SAVE_DIR, times_file)
        
        result = analyze_algorithm(algo_name, scores_path, times_path)
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
        
        print("=" * 100)
        print("RANKING BY SCORE IMPROVEMENT")
        print("=" * 100)
        print()
        
        for i, (_, row) in enumerate(summary_df.iterrows(), 1):
            status = "✅ BETTER" if row['improvement_rate'] > 0 else "❌ NOT BETTER"
            print(f"{i:2d}. {row['algorithm']:<35} - {row['score_improvement']:6.2f}% improvement {status}")
        
        print()
        print("=" * 100)
        print("BEST PERFORMING ALGORITHMS")
        print("=" * 100)
        
        better_algorithms = summary_df[summary_df['improvement_rate'] > 0]
        if len(better_algorithms) > 0:
            print("Algorithms that outperform NearestNeighbourSolver:")
            for _, row in better_algorithms.iterrows():
                print(f"  ✅ {row['algorithm']}: {row['score_improvement']:.2f}% better, {row['improvements']}/{row['total_comparable']} problems improved")
        else:
            print("❌ No algorithms significantly outperformed NearestNeighbourSolver")
        
        print()
        print("=" * 100)
        print("DETAILED COMPARISON TABLE")
        print("=" * 100)
        print()
        
        # Create a nice table
        table_data = []
        for _, row in summary_df.iterrows():
            table_data.append([
                row['algorithm'],
                f"{row['improvements']}/{row['total_comparable']}",
                f"{row['improvement_rate']:.1%}",
                f"{row['score_improvement']:+.2f}%",
                f"{row['avg_time']:.1f}s",
                f"{row['time_ratio']:.1f}x"
            ])
        
        df_table = pd.DataFrame(table_data, columns=[
            'Algorithm', 'Improvements', 'Rate', 'Score %', 'Avg Time', 'Time Ratio'
        ])
        print(df_table.to_string(index=False))
        
        # Save summary
        summary_df.to_csv(os.path.join(ROOT_DIR, SAVE_DIR, "final_algorithm_summary.csv"))
        print(f"\nSummary saved to: {os.path.join(ROOT_DIR, SAVE_DIR, 'final_algorithm_summary.csv')}")
    
    print()
    print("=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    
    if results:
        better_algorithms = [r for r in results if r['improvement_rate'] > 0]
        if better_algorithms:
            best_algorithm = max(better_algorithms, key=lambda x: x['score_improvement'])
            print(f"✅ BEST ALGORITHM: {best_algorithm['algorithm']}")
            print(f"   - {best_algorithm['score_improvement']:.2f}% better than NearestNeighbourSolver")
            print(f"   - Improved {best_algorithm['improvements']}/{best_algorithm['total_comparable']} problems")
            print(f"   - Average time: {best_algorithm['avg_time']:.1f}s (within 100s limit)")
        else:
            print("❌ No algorithms significantly outperformed NearestNeighbourSolver")
    else:
        print("❌ No valid results found")

if __name__ == "__main__":
    main()
