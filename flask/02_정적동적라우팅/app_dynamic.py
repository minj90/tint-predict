from flask import Flask
app = Flask(__name__) #앱 인스턴스 생성

@app.route("/")
def home() :
    return "<h1>Hello world : home</h1>"

@app.route("/profile/<username>") # url을 "<변수>"형태로 사용
def get_profile(username):
    return "username? : " + username


if __name__ =="__main__":
    app.run()