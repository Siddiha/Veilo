import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'image_picker_controller.dart';
import 'image_preview.dart';

class ImageUploadPage extends StatelessWidget {
  const ImageUploadPage({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => ImagePickerController(),
      child: Consumer<ImagePickerController>(
        builder: (context, ctrl, _) {
          return Scaffold(
            appBar: AppBar(title: const Text('Image Upload & Analysis')),
            body: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  if (ctrl.selectedFile != null)
                    Expanded(child: ImagePreview(file: ctrl.selectedFile as File))
                  else
                    const Expanded(
                      child: Center(child: Text('No image selected')),
                    ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      ElevatedButton(
                        onPressed: ctrl.pickImage,
                        child: const Text('Pick Image'),
                      ),
                      const SizedBox(width: 12),
                      ElevatedButton(
                        onPressed: ctrl.selectedFile == null ? null : () {
                          // TODO: call backend to analyze
                        },
                        child: const Text('Analyze'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}


