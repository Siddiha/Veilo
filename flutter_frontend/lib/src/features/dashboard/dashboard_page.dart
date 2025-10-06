import 'package:flutter/material.dart';
import 'results_summary.dart';

class DashboardPage extends StatelessWidget {
  final double probability;
  const DashboardPage({super.key, required this.probability});

  @override
  Widget build(BuildContext context) {
    final recommendation = probability >= 0.66
        ? 'Consult a doctor immediately'
        : (probability >= 0.33 ? 'Consider medical consultation' : 'Continue monitoring');
    return Scaffold(
      appBar: AppBar(title: const Text('Results Dashboard')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: ResultsSummary(probability: probability, recommendation: recommendation),
      ),
    );
  }
}


