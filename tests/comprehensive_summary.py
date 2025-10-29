import pandas as pd
import os
import numpy as np

def load_results(filename):
    """Load results from CSV file"""
    try:
        df = pd.read_csv(f"evaluation_results/{filename}")
        return df
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None

def main():
    print("=== COMPREHENSIVE TSP SOLVER EVALUATION SUMMARY ===\n")
    
    # List of all algorithms tested with actual filenames
    algorithms = [
        ("TwoOptSolver", "2-opt local search", "two_opt_scores.csv", "two_opt_run_times.csv"),
        ("ThreeOptSolver", "3-opt local search", "three_opt_scores.csv", "three_opt_run_times.csv"),
        ("SimulatedAnnealingSolver", "Simulated Annealing", "simulated_annealing_scores.csv", "simulated_annealing_run_times.csv"),
        ("GeneticAlgorithmSolver", "Basic Genetic Algorithm", "genetic_algorithm_scores.csv", "genetic_algorithm_run_times.csv"),
        ("CheapestInsertionTwoOptSolver", "Cheapest Insertion + 2-opt", "cheapestinsertiontwoopt_score.csv", "cheapestinsertiontwoopt_run_times.csv"),
        ("ChristofidesSolver", "Christofides approximation", "christofides_scores.csv", "christofides_run_times.csv"),
        ("NearestInsertionTwoOptSolver", "Nearest Insertion + 2-opt", "nearest_insertion_two_opt_scores.csv", "nearest_insertion_two_opt_run_times.csv"),
        ("FarthestInsertionTwoOptSolver", "Farthest Insertion + 2-opt", "farthest_insertion_two_opt_scores.csv", "farthest_insertion_two_opt_run_times.csv"),
        ("LinKernighanSolver", "Lin-Kernighan heuristic", "lin_kernighan_scores.csv", "lin_kernighan_run_times.csv"),
        ("AdvancedGeneticSolver", "Advanced Genetic Algorithm", "advanced_genetic_scores.csv", "advanced_genetic_run_times.csv"),
        ("AdvancedSimulatedAnnealingSolver", "Advanced Simulated Annealing", "advanced_simulated_annealing_scores.csv", "advanced_simulated_annealing_run_times.csv"),
        ("AntColonySolver", "Ant Colony Optimization", "ant_colony_scores.csv", "ant_colony_run_times.csv"),
        ("VariableNeighborhoodSolver", "Variable Neighborhood Search", "variable_neighborhood_scores.csv", "variable_neighborhood_run_times.csv"),
        ("MemeticSolver", "Memetic Algorithm (GA + Local Search)", "memetic_scores.csv", "memetic_run_times.csv"),
        ("HybridGeneticSolver", "Hybrid Genetic Algorithm", "hybrid_genetic_scores.csv", "hybrid_genetic_run_times.csv")
    ]
    
    results = []
    
    for algo_name, description, scores_file, times_file in algorithms:
        scores_df = load_results(scores_file)
        times_df = load_results(times_file)
        
        if scores_df is not None and times_df is not None:
            # Get algorithm and NN scores
            if algo_name in scores_df.columns and 'NearestNeighbourSolver' in scores_df.columns:
                algo_scores = scores_df[algo_name].values
                nn_scores = scores_df['NearestNeighbourSolver'].values
                
                # Calculate improvements
                improvements = sum(1 for a, n in zip(algo_scores, nn_scores) 
                                 if a != float('inf') and n != float('inf') and a < n)
                comparable = sum(1 for a, n in zip(algo_scores, nn_scores) 
                               if a != float('inf') and n != float('inf'))
                
                # Calculate averages
                avg_algo = np.mean([s for s in algo_scores if s != float('inf')])
                avg_nn = np.mean([s for s in nn_scores if s != float('inf')])
                
                # Calculate average run time
                avg_time = np.mean([t for t in times_df[algo_name].values if t != float('inf')])
                
                # Determine status
                if improvements > 0:
                    status = "✅ BETTER"
                else:
                    status = "❌ NOT BETTER"
                
                results.append({
                    'Algorithm': algo_name,
                    'Description': description,
                    'Improvements': f"{improvements}/{comparable}",
                    'Avg Score': f"{avg_algo:.0f}",
                    'Avg NN Score': f"{avg_nn:.0f}",
                    'Avg Time (s)': f"{avg_time:.1f}",
                    'Status': status
                })
            else:
                results.append({
                    'Algorithm': algo_name,
                    'Description': description,
                    'Improvements': "N/A",
                    'Avg Score': "N/A",
                    'Avg NN Score': "N/A",
                    'Avg Time (s)': "N/A",
                    'Status': "❌ NO DATA"
                })
        else:
            results.append({
                'Algorithm': algo_name,
                'Description': description,
                'Improvements': "N/A",
                'Avg Score': "N/A",
                'Avg NN Score': "N/A",
                'Avg Time (s)': "N/A",
                'Status': "❌ NO DATA"
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Save to CSV
    summary_df.to_csv("evaluation_results/comprehensive_algorithm_summary.csv", index=False)
    
    # Print summary
    print("ALGORITHM PERFORMANCE SUMMARY:")
    print("=" * 100)
    print(f"{'Algorithm':<25} {'Improvements':<12} {'Avg Score':<12} {'Avg NN':<12} {'Time(s)':<10} {'Status'}")
    print("-" * 100)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Algorithm']:<25} {row['Improvements']:<12} {row['Avg Score']:<12} {row['Avg NN Score']:<12} {row['Avg Time (s)']:<10} {row['Status']}")
    
    # Count successful algorithms
    successful = summary_df[summary_df['Status'] == '✅ BETTER']
    print(f"\nSUCCESSFUL ALGORITHMS: {len(successful)}/{len(summary_df)}")
    
    if len(successful) > 0:
        print("\nBEST PERFORMING ALGORITHMS:")
        for _, row in successful.iterrows():
            print(f"  • {row['Algorithm']}: {row['Improvements']} improvements, avg score {row['Avg Score']}")
    
    print(f"\nSummary saved to: evaluation_results/comprehensive_algorithm_summary.csv")

if __name__ == "__main__":
    main()
