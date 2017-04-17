# Sim-to-Real
From simulation to real world using deep generative models


Exp swimmer, random moves, 1 000 000 samples:
2 hidden layers with 128, history  = 1, batchnorm
after 100epochs
test_percent_error: 0.1347082406282425
test_squared_error: 0.13742806017398834
train_percent_error: 0.12277309596538544
train_squared_error: 0.1374274343252182

Exp swimmer, random moves, 1 000 000 samples:
2 hidden layers with 128, history  = 2, batchnorm
after 102epochs
test_percent_error: 0.11568228155374527
test_squared_error: 0.13852712512016296
train_percent_error: 0.1286197453737259
train_squared_error: 0.13872990012168884

Exp swimmer, random moves, 1 000 000 samples:
2 hidden layers with 256, history = 2, batchnorm
after 102epochs
test_percent_error: 0.11568228155374527
test_squared_error: 0.13852712512016296
train_percent_error: 0.1286197453737259
train_squared_error: 0.13872990012168884

Exp swimmer, random moves, 1 000 000 samples:
2 hidden layers with 128, history = 3, batchnorm
after 96epochs
test_percent_error: 0.11686143279075623
test_squared_error: 0.1399289220571518
train_percent_error: 0.12781818211078644
train_squared_error: 0.1390911340713501

Exp swimmer, random moves, 1 000 000 samples:
2 hidden layers with 256, history = 3, batchnorm
after 116epochs
test_percent_error: 0.1150
test_squared_error: 0.1399
train_percent_error: 0.1230
train_squared_error: 0.139
