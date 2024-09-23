import turtle
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Função para re-treinar o modelo
def retrain_model(game_data, model_filename='pong_model.pkl'):
    df = pd.read_csv('game_data.csv')

    # Definir X (features) e y (target)
    X = df[['ball_x', 'ball_y', 'left_pad_y',
             'ball_dx', 'ball_dy', 'left_pad_distance',
             'ball_distance_to_left_pad', 'ball_direction']]
    y = df['player_action']

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar o modelo usando XGBoost
    model = xgb.XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train, y_train)

    # Avaliar o modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

    # Salvar o modelo treinado
    joblib.dump(model, model_filename)
    joblib.dump(encoder, 'encoder.pkl')
    print("Modelo re-treinado e salvo com sucesso!")
    return model

model = joblib.load('pong_model.pkl')
encoder = joblib.load('encoder.pkl')
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
sketch.write("Jogador da Esquerda : 0    Jogador da Direita: 0",
             align="center", font=("Courier", 24, "normal"))

left_player_action = "none"

def paddleaup():
    global left_player_action
    y = left_pad.ycor()
    if y < 250:  # Limit paddle movement
        y += 20
        left_pad.sety(y)
    left_player_action = "up"
        


def paddleadown():
    global left_player_action
    y = left_pad.ycor()
    if y > -240:  # Limit paddle movement
        y -= 20
        left_pad.sety(y)
    left_player_action = "down"


def paddlebup():
    y = right_pad.ycor()
    if y < 250:  # Limit paddle movement
        y += 20
        right_pad.sety(y)


def paddlebdown():
    y = right_pad.ycor()
    if y > -240:  # Limit paddle movement
        y -= 20
        right_pad.sety(y)


# Keyboard bindings
sc.listen()
sc.onkeypress(paddleaup, "w")
sc.onkeypress(paddleadown, "s")
sc.onkeypress(paddlebup, "Up")
sc.onkeypress(paddlebdown, "Down")

# Definir o número de jogadas antes de re-treinar
retrain_interval = 1000
current_plays = 0

# Inicializar uma lista para armazenar dados do jogo
game_data = []

# Função para registrar o estado do jogo
def record_game_data(ball_x, ball_y, left_pad_y, right_pad_y, action, ball_dx, ball_dy):
    left_pad_distance = ball_x + 400
    right_pad_distance = 400 - ball_x

    # Novas features
    ball_distance_to_left_pad = abs(ball_x + 400 - left_pad_y)
    ball_direction = 1 if ball_dx > 0 else -1  # 1 para direita, -1 para esquerda

    if ball_distance_to_left_pad < 150:
        game_data.append({
            'ball_x': ball_x,
            'ball_y': ball_y,
            'left_pad_y': left_pad_y,
            'player_action': action,
            'ball_dx': ball_dx,
            'ball_dy': ball_dy,
            'left_pad_distance': left_pad_distance,
            'ball_distance_to_left_pad': ball_distance_to_left_pad,
            'ball_direction': ball_direction,
        })

def save_game_data_to_csv():
    import csv

    # Abra o arquivo em modo de append
    with open('game_data.csv', mode='a', newline='') as file:
        fieldnames = ['ball_x', 'ball_y', 'left_pad_y',
                      'player_action', 'ball_dx', 'ball_dy', 
                      'left_pad_distance',
                      'ball_distance_to_left_pad', 'ball_direction']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Escreva o cabeçalho apenas se o arquivo estiver vazio
        if file.tell() == 0:
            writer.writeheader()

        # Escreva os dados do jogo
        for data in game_data:
            writer.writerow(data)

    print("Dados do jogo salvos em game_data.csv")
  

# Função para usar o modelo para prever a ação com base no estado atual do jogo
def predict_action(ball_x, ball_y, left_pad_y, right_pad_y, ball_dx, ball_dy):
    left_pad_distance = ball_x + 400
    right_pad_distance = 400 - ball_x
    ball_distance_to_left_pad = abs(ball_x + 400 - left_pad_y)
    ball_direction = 1 if ball_dx > 0 else -1
    future_ball_x = ball_x + ball_dx * 10
    future_ball_y = ball_y + ball_dy * 10
    distance_to_goal = abs(future_ball_x - 400)

    input_data = pd.DataFrame({
        'ball_x': [ball_x],
        'ball_y': [ball_y],
        'left_pad_y': [left_pad_y],
        'ball_dx': [ball_dx],
        'ball_dy': [ball_dy],
        'left_pad_distance': [left_pad_distance],
        'ball_distance_to_left_pad': [ball_distance_to_left_pad],
        'ball_direction': [ball_direction],
    })
    action_pred = model.predict(input_data)[0]
    return encoder.inverse_transform([action_pred])[0]

# Main game loop
while True:
    sc.update()
    time.sleep(0.001)  # Add delay to make game smoother

    hit_ball.setx(hit_ball.xcor() + hit_ball.dx)
    hit_ball.sety(hit_ball.ycor() + hit_ball.dy)

    # Predict the action of the right player
    ai_player_action = predict_action(
        hit_ball.xcor(),
        hit_ball.ycor(),
        left_pad.ycor(),
        right_pad.ycor(),
        hit_ball.dx,
        hit_ball.dy
    )

    if ai_player_action == "up":
        paddleaup()
    elif ai_player_action == "down":
        paddleadown()

    print('asd', left_player_action)
    # if left_player_action != 'none': 
    record_game_data(
        hit_ball.xcor(),
        hit_ball.ycor(),
        left_pad.ycor(),
        right_pad.ycor(),
        left_player_action,  # Ação do jogador
        hit_ball.dx,
        hit_ball.dy
    )

    left_player_action = "none"

    current_plays += 1

    # Verificar se é hora de re-treinar o modelo
    if current_plays % retrain_interval == 0:
        print(f"Re-treinando modelo após {current_plays} jogadas...")
        # save_game_data_to_csv()  # Salvar os dados em CSV
        # model = retrain_model(game_data)  # Re-treinar o modelo
        current_plays = 0  # Resetar contagem de jogadas


    # Checking borders
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
        sketch.write("Jogador da Esquerda: {}    Jogador da Direita: {}".format(
            left_player, right_player), align="center",
            font=("Courier", 24, "normal"))

    if hit_ball.xcor() < -500:
        hit_ball.goto(0, 0)
        hit_ball.dy *= -1
        right_player += 1
        sketch.clear()
        sketch.write("Jogador da Esquerda: {}    Jogador da Direita: {}".format(
            left_player, right_player), align="center",
            font=("Courier", 24, "normal"))
        

    if left_player == max_points or right_player == max_points:
      print("Fim do jogo!")
      # save_game_data_to_csv()  # Salvar os dados em CSV
      break  # Sair do loop principal para encerrar o jogo

    # Paddle ball collision
    if (hit_ball.xcor() > 360 and hit_ball.xcor() < 370) and \
            (hit_ball.ycor() < right_pad.ycor() + 70 and hit_ball.ycor() > right_pad.ycor() - 70):
        hit_ball.setx(360)
        hit_ball.dx *= -1

    if (hit_ball.xcor() < -360 and hit_ball.xcor() > -370) and \
            (hit_ball.ycor() < left_pad.ycor() + 70 and hit_ball.ycor() > left_pad.ycor() - 70):
        hit_ball.setx(-360)
        hit_ball.dx *= -1
  
