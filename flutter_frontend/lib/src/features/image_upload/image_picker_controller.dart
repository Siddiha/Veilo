import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';

class ImagePickerController extends ChangeNotifier {
  final ImagePicker _picker = ImagePicker();
  File? selectedFile;

  Future<void> pickImage() async {
    final XFile? file = await _picker.pickImage(source: ImageSource.gallery);
    if (file != null) {
      selectedFile = File(file.path);
      notifyListeners();
    }
  }
}


