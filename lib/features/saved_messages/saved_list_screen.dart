import 'package:flutter/material.dart';

class SavedListScreen extends StatelessWidget {
  static const String routeName = '/saved';
  const SavedListScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Mensajes guardados'),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: ListView(
          children: const [
            SizedBox(height: 20),
            // Tarjeta 1
            Card(
              child: ListTile(
                leading: CircleAvatar(
                  backgroundColor: Colors.grey,
                  child: Icon(Icons.play_arrow, color: Colors.white),
                ),
                title: Text('Hola, ¿cómo estás?'),
                subtitle: Text('Guardado hace 2 días'),
                trailing: Icon(Icons.arrow_forward_ios),
              ),
            ),
            SizedBox(height: 16),
            // Tarjeta 2
            Card(
              child: ListTile(
                leading: CircleAvatar(
                  backgroundColor: Colors.grey,
                  child: Icon(Icons.play_arrow, color: Colors.white),
                ),
                title: Text('Gracias por tu ayuda'),
                subtitle: Text('Guardado hace 1 día'),
                trailing: Icon(Icons.arrow_forward_ios),
              ),
            ),
          ],
        ),
      ),
    );
  }
}