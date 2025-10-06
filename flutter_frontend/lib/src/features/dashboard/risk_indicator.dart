import 'package:flutter/material.dart';

class RiskIndicator extends StatelessWidget {
  final double score; // 0..1
  const RiskIndicator({super.key, required this.score});

  Color get color => score >= 0.66
      ? Colors.red
      : (score >= 0.33 ? Colors.orange : Colors.green);

  String get label => score >= 0.66
      ? 'High'
      : (score >= 0.33 ? 'Medium' : 'Low');

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        CircleAvatar(backgroundColor: color, radius: 8),
        const SizedBox(width: 8),
        Text('$label risk (${(score * 100).toStringAsFixed(0)}%)'),
      ],
    );
  }
}


