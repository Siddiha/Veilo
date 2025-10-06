import 'question_model.dart';

double calculateRisk(QuestionnaireModel q) {
  double score = 0;
  score += (q.age / 100.0).clamp(0, 0.3);
  score += ((q.smokingYears * q.packsPerDay) / 100.0).clamp(0, 0.4);
  score += q.familyHistory ? 0.1 : 0.0;
  score += (q.symptoms.length * 0.05).clamp(0, 0.2);
  return score.clamp(0, 1);
}


