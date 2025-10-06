import 'package:flutter/material.dart';
import 'risk_indicator.dart';

class ResultsSummary extends StatelessWidget {
  final double probability;
  final String recommendation;
  const ResultsSummary({super.key, required this.probability, required this.recommendation});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        RiskIndicator(score: probability),
        const SizedBox(height: 8),
        Text(recommendation),
        const SizedBox(height: 8),
        const Text(
          'Disclaimer: This is not a medical diagnosis. Consult a healthcare professional.',
          style: TextStyle(fontSize: 12, color: Colors.black54),
        ),
      ],
    );
  }
}


