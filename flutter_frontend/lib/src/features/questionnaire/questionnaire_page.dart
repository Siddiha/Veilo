import 'package:flutter/material.dart';
import 'question_model.dart';
import 'risk_calculator.dart';

class QuestionnairePage extends StatefulWidget {
  const QuestionnairePage({super.key});

  @override
  State<QuestionnairePage> createState() => _QuestionnairePageState();
}

class _QuestionnairePageState extends State<QuestionnairePage> {
  final _formKey = GlobalKey<FormState>();
  int age = 30;
  String gender = 'Male';
  int smokingYears = 0;
  double packsPerDay = 0;
  bool familyHistory = false;
  final Set<String> symptoms = {};

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Health Questionnaire')),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            TextFormField(
              decoration: const InputDecoration(labelText: 'Age'),
              initialValue: age.toString(),
              keyboardType: TextInputType.number,
              onChanged: (v) => age = int.tryParse(v) ?? age,
            ),
            DropdownButtonFormField(
              initialValue: gender,
              items: const [
                DropdownMenuItem(value: 'Male', child: Text('Male')),
                DropdownMenuItem(value: 'Female', child: Text('Female')),
              ],
              onChanged: (v) => setState(() => gender = v as String),
              decoration: const InputDecoration(labelText: 'Gender'),
            ),
            TextFormField(
              decoration: const InputDecoration(labelText: 'Smoking years'),
              initialValue: smokingYears.toString(),
              keyboardType: TextInputType.number,
              onChanged: (v) => smokingYears = int.tryParse(v) ?? smokingYears,
            ),
            TextFormField(
              decoration: const InputDecoration(labelText: 'Packs per day'),
              initialValue: packsPerDay.toString(),
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              onChanged: (v) => packsPerDay = double.tryParse(v) ?? packsPerDay,
            ),
            SwitchListTile(
              value: familyHistory,
              onChanged: (v) => setState(() => familyHistory = v),
              title: const Text('Family history of cancer'),
            ),
            const SizedBox(height: 8),
            const Text('Symptoms'),
            Wrap(
              spacing: 8,
              children: [
                for (final s in ['Cough', 'Chest pain', 'Breathing issues'])
                  FilterChip(
                    label: Text(s),
                    selected: symptoms.contains(s),
                    onSelected: (sel) => setState(() {
                      if (sel)
                        symptoms.add(s);
                      else
                        symptoms.remove(s);
                    }),
                  )
              ],
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () {
                final model = QuestionnaireModel(
                  age: age,
                  gender: gender,
                  smokingYears: smokingYears,
                  packsPerDay: packsPerDay,
                  symptoms: symptoms.toList(),
                  familyHistory: familyHistory,
                );
                final score = calculateRisk(model);
                final level =
                    score >= 0.66 ? 'High' : (score >= 0.33 ? 'Medium' : 'Low');
                showDialog(
                  context: context,
                  builder: (_) => AlertDialog(
                    title: const Text('Risk Score'),
                    content: Text(
                        'Score: ${score.toStringAsFixed(2)}\nLevel: $level'),
                  ),
                );
              },
              child: const Text('Calculate Risk'),
            ),
          ],
        ),
      ),
    );
  }
}
