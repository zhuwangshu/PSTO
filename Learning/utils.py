import numpy as np


def angle2signal(angles):
  assert len(angles) == 8
  signals = [0]*8
  # signals[0] = int(np.clip(330   + angles[0] * 150.0 / 90, 255, 405))
  # signals[1] = int(np.clip(342.5 + angles[1] * 155.0 / 90, 265, 420))
  # signals[2] = int(np.clip(340   + angles[2] * 160.0 / 90, 260, 420))
  # signals[3] = int(np.clip(342.5 + angles[3] * 155.0 / 90, 265, 420))
  # signals[4] = int(np.clip(260   + angles[4] * 160.0 / 90, 180, 340))
  # signals[5] = int(np.clip(257.5 + angles[5] * 155.0 / 90, 180, 335))
  # signals[6] = int(np.clip(265   + angles[6] * 160.0 / 90, 185, 345))
  # signals[7] = int(np.clip(275   + angles[7] * 160.0 / 90, 195, 355))
  multipler = 8  # 8
  signals[0] = int(np.clip( angles[0] * multipler, -30, 30))
  signals[1] = int(np.clip( angles[1] * multipler, -30, 30))
  signals[2] = int(np.clip( angles[2] * multipler, -30, 30))
  signals[3] = int(np.clip( angles[3] * multipler, -30, 30))
  signals[4] = int(np.clip( angles[4] * multipler, -30, 30))
  signals[5] = int(np.clip( angles[5] * multipler, -30, 30))
  signals[6] = int(np.clip( angles[6] * multipler, -30, 30))
  signals[7] = int(np.clip( angles[7] * multipler, -30, 30))
  return np.array(signals)


def signal2angle(signals):  # until now, not used
  assert len(signals) == 8
  angles = [45]*8
  angles[0] = np.clip((signals[0] - 330  ) / 150.0 * 90, 0, 90)
  angles[1] = np.clip((signals[1] - 342.5) / 155.0 * 90, 0, 90)
  angles[2] = np.clip((signals[2] - 340  ) / 160.0 * 90, 0, 90)
  angles[3] = np.clip((signals[3] - 342.5) / 155.0 * 90, 0, 90)
  angles[4] = np.clip((signals[4] - 260  ) / 160.0 * 90, 0, 90)
  angles[5] = np.clip((signals[5] - 257.5) / 155.0 * 90, 0, 90)
  angles[6] = np.clip((signals[6] - 265  ) / 160.0 * 90, 0, 90)
  angles[7] = np.clip((signals[7] - 275  ) / 160.0 * 90, 0, 90)
  return np.array(angles)

def observationRegularization(signals): 
  assert len(signals) == 16
  angles = [0.0]*16
  divider = 30     #30
  divider2 = 0.1
  mid = 330
  angles[0] = np.clip((signals[0] - mid  ) / divider , -1, 1)
  angles[1] = np.clip((signals[1] - mid) / divider , -1, 1)
  angles[2] = np.clip((signals[2] - mid  ) / divider , -1, 1)
  angles[3] = np.clip((signals[3] - mid) / divider , -1, 1)
  angles[4] = np.clip((signals[4] - mid  ) / divider , -1, 1)
  angles[5] = np.clip((signals[5] - mid) / divider, -1, 1)
  angles[6] = np.clip((signals[6] - mid  ) / divider , -1, 1)
  angles[7] = np.clip((signals[7] - mid  ) / divider , -1, 1)
  angles[8] = np.clip((signals[8] ) / divider2 , -10, 10)
  angles[9] = np.clip((signals[9] ) / divider2 , -10, 10)
  angles[10] = np.clip((signals[10] ) / divider2 , -10, 10)
  angles[11] = np.clip((signals[11] ) / divider2 , -10, 10)
  angles[12] = np.clip((signals[12] ) / divider2 , -10, 10)
  angles[13] = np.clip((signals[13] ) / divider2 , -10, 10)
  angles[14] = np.clip((signals[14] ) / divider2 , -10, 10)
  angles[15] = np.clip((signals[15] ) / divider2 , -10, 10)
  return np.array(angles)