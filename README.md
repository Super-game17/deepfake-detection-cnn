# deepfake-detection-cnn

Application Streamlit de détection de deepfakes avec détection de visages SCRFD,
classification ResNet-50 et explication Grad-CAM.

## Lancer localement

Depuis la racine du projet :

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run src\app.py
```

Le fichier `best_model.pth` local est utilisé s'il est présent. Sinon,
l'application le télécharge automatiquement depuis Hugging Face au premier
lancement. L'URL peut être remplacée avec la variable d'environnement
`MODEL_URL`.

## Déployer sur Streamlit Community Cloud

1. Pousser le dépôt GitHub sans `best_model.pth` : les fichiers `*.pth` sont
   volontairement ignorés par `.gitignore`.
2. Dans Streamlit Community Cloud, choisir le dépôt, la branche et le fichier
   principal `src/app.py`.
3. Sélectionner Python 3.12 si l'option est disponible.
4. Le fichier `requirements.txt` installe la version CPU de ONNX Runtime,
   adaptée à une instance Streamlit sans GPU.
5. Au premier démarrage, l'application télécharge automatiquement le modèle
   public de 94 Mo depuis Hugging Face. Les démarrages suivants réutilisent le
   fichier présent dans l'instance.

Le modèle est public et ne nécessite donc pas de secret Hugging Face. Si le
modèle devient privé, définir `MODEL_URL` avec une URL authentifiée et utiliser
un secret Streamlit plutôt que de mettre un jeton dans le code.
