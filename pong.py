import turtle
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def retrain_model(game_data, model_filename='pong_model_rf.pkl'):
    df = pd.read_csv('game_data.csv')

    X = df[['ball_x', 'ball_y', 'ball_dx', 'ball_dy']]
    y = df['left_pad_y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Erro quadrático médio do modelo: {mse:.2f}")

    joblib.dump(model, model_filename)
    print("Modelo re-treinado e salvo com sucesso!")
    return model

model = joblib.load('pong_model_rf.pkl')
print("Modelo carregado com sucesso!")

# Create screen
sc = turtle.Screen()
sc.title("Pong game")
sc.bgcolor("white")
sc.setup(width=1000, height=600)

# Left paddle
left_pad = turtle.Turtle()
left_pad.speed(0)
left_pad.shape("square")
left_pad.color("black")
left_pad.shapesize(stretch_wid=6, stretch_len=2)
left_pad.penup()
left_pad.goto(-400, 0)

# Right paddle
right_pad = turtle.Turtle()
right_pad.speed(0)
right_pad.shape("square")
right_pad.color("black")
right_pad.shapesize(stretch_wid=6, stretch_len=2)
right_pad.penup()
right_pad.goto(400, 0)

# Ball of circle shape
hit_ball = turtle.Turtle()
hit_ball.speed(4)  # Adjusted speed
hit_ball.shape("circle")
hit_ball.color("blue")
hit_ball.penup()
hit_ball.goto(0, 0)
hit_ball.dx = 5
hit_ball.dy = -5

# Initialize the score
left_player = 0
right_player = 0
max_points = 50

# Displays the score
sketch = turtle.Turtle()
sketch.speed(0)
sketch.color("blue")
sketch.penup()
sketch.hideturtle()
sketch.goto(0, 260)
sketch.write("Jogador IA : 0    Jogador da Direita: 0",
             align="center", font=("Courier", 24, "normal"))

left_player_action = "none"

def paddleaup():
    global left_player_action
    y = left_pad.ycor()
    if y < 250:
        y += 20
        left_pad.sety(y)
    left_player_action = "up"
        


def paddleadown():
    global left_player_action
    y = left_pad.ycor()
    if y > -240:
        y -= 20
        left_pad.sety(y)
    left_player_action = "down"


def paddlebup():
    y = right_pad.ycor()
    if y < 250:
        y += 20
        right_pad.sety(y)


def paddlebdown():
    y = right_pad.ycor()
    if y > -240:
        y -= 20
        right_pad.sety(y)


# Keyboard bindings
sc.listen()
sc.onkeypress(paddleaup, "w")
sc.onkeypress(paddleadown, "s")
sc.onkeypress(paddlebup, "Up")
sc.onkeypress(paddlebdown, "Down")

retrain_interval = 1000
current_plays = 0

game_data = []

def record_game_data(ball_x, ball_y, left_pad_y, ball_dx, ball_dy):

    game_data.append({
        'ball_x': ball_x,
        'ball_y': ball_y,
        'ball_dx': ball_dx,
        'ball_dy': ball_dy,
        'left_pad_y': left_pad_y,
    })

def save_game_data_to_csv():
    import csv

    with open('game_data.csv', mode='a', newline='') as file:
        fieldnames = ['ball_x', 'ball_y', 'ball_dx', 'ball_dy','left_pad_y']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if file.tell() == 0:
            writer.writeheader()

        for data in game_data:
            writer.writerow(data)

    print("Dados do jogo salvos em game_data.csv")

def predict_paddle_position(ball_x, ball_y, ball_dx, ball_dy):
    input_data = pd.DataFrame({
        'ball_x': [ball_x],
        'ball_y': [ball_y],
        'ball_dx': [ball_dx],
        'ball_dy': [ball_dy]
    })

    predicted_y = model.predict(input_data)[0]
    return predicted_y

while True:
    sc.update()
    time.sleep(0.001)

    hit_ball.setx(hit_ball.xcor() + hit_ball.dx)
    hit_ball.sety(hit_ball.ycor() + hit_ball.dy)

    predicted_y = predict_paddle_position(
        hit_ball.xcor(),
        hit_ball.ycor(),
        hit_ball.dx,
        hit_ball.dy
    )

    left_pad.sety(predicted_y)

    # record_game_data(
    #     hit_ball.xcor(),
    #     hit_ball.ycor(),
    #     left_pad.ycor(),
    #     hit_ball.dx,
    #     hit_ball.dy
    # )


    current_plays += 1

    # if current_plays % retrain_interval == 0:
    #     print(f"Retreinando modelo após {current_plays} jogadas...")
    #     save_game_data_to_csv()
    #     model = retrain_model(game_data)  
    #     current_plays = 0


    if hit_ball.ycor() > 280:
        hit_ball.sety(280)
        hit_ball.dy *= -1

    if hit_ball.ycor() < -280:
        hit_ball.sety(-280)
        hit_ball.dy *= -1

    if hit_ball.xcor() > 500:
        hit_ball.goto(0, 0)
        hit_ball.dy *= -1
        left_player += 1
        sketch.clear()
        sketch.write("Jogador IA: {}    Jogador da Direita: {}".format(
            left_player, right_player), align="center",
            font=("Courier", 24, "normal"))

    if hit_ball.xcor() < -500:
        hit_ball.goto(0, 0)
        hit_ball.dy *= -1
        right_player += 1
        sketch.clear()
        sketch.write("Jogador IA: {}    Jogador da Direita: {}".format(
            left_player, right_player), align="center",
            font=("Courier", 24, "normal"))
        

    if left_player == max_points or right_player == max_points:
      print("Fim do jogo!")
      break

    # Paddle ball collision
    if (hit_ball.xcor() > 360 and hit_ball.xcor() < 370) and \
            (hit_ball.ycor() < right_pad.ycor() + 70 and hit_ball.ycor() > right_pad.ycor() - 70):
        hit_ball.setx(360)
        hit_ball.dx *= -1

    if (hit_ball.xcor() < -360 and hit_ball.xcor() > -370) and \
            (hit_ball.ycor() < left_pad.ycor() + 70 and hit_ball.ycor() > left_pad.ycor() - 70):
        hit_ball.setx(-360)
        hit_ball.dx *= -1
  
