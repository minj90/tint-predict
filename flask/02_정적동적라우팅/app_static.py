from flask import Flask
app = Flask(__name__) #앱 인스턴스 생성

@app.route("/")
def home() :
    return "<h1>Hello world : home</h1>"

@app.route("/hello") #http://127.0.0.1:5000/hello
def hello() :
    return "<h1>Hello world : hello</h1>"

if __name__ == "__main__": #메인코드라면
    app.run()
    