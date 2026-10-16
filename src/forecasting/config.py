import pandas as pd

HORIZON = 28  # dias a prever
TEST_START = pd.Timestamp("2016-04-25")  # início da janela de teste final
N_VAL_WINDOWS = 4  # janelas de validação antes do teste

MIN_RATE_FOR_STOCKOUT = 2.0  # vendas médias/dia a partir das quais um produto "vende sempre"
MIN_ZERO_RUN = 14  # dias seguidos a zero para contar como indisponível
