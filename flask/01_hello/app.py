from flask import Flask
app = Flask(__name__) #앱 인스턴스 생성

@app.route("/")
def hello():
    return "<h1>Hello world</h1>"


if __name__ == "__main__":
    app.run()

