# Cargamos las librerias necesarias
from flask import Flask
from app import views

# Gateway interprete usado para el web server 
app = Flask(__name__)

# Rutas de navegación del backend
app.add_url_rule(rule='/', 
                 endpoint='home', 
                 view_func=views.index)
app.add_url_rule(rule='/app/', 
                 endpoint='app', 
                 view_func=views.app)
app.add_url_rule(rule='/app/gender/', 
                 endpoint='gender', 
                 view_func=views.genderapp, 
                 methods=['GET', 'POST'])

# metodo principal que ejecuta el aplicativo en el puerto 3000
if __name__ == "__main__":
    app.run(debug=True, port=3000)