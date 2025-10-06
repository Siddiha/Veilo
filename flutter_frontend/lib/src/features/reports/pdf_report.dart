import 'dart:typed_data';
import 'package:pdf/widgets.dart' as pw;

Future<Uint8List> buildPdfReport({
  required String title,
  required String summary,
}) async {
  final pdf = pw.Document();
  pdf.addPage(
    pw.Page(
      build: (context) => pw.Column(children: [
        pw.Text(title, style: pw.TextStyle(fontSize: 24)),
        pw.SizedBox(height: 8),
        pw.Text(summary),
      ]),
    ),
  );
  return pdf.save();
}


