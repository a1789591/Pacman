
from captureAgents import CaptureAgent
import random, time, util
# from game import Directions
import game

from agents.sample.baselineTeam import PositionSearchProblem 
from util import Queue
from collections import Counter

CD = os.path.dirname(os.path.abspath(__file__))

FF_EXECUTABLE_PATH = f"{CD}/1415562/ff"
Packman_pddl = f"{CD}/pacman"