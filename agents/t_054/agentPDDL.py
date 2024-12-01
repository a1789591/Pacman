import time
import util
import os
import re
import subprocess
from util import Queue
from collections import Counter

CD = os.path.dirname(os.path.abspath(__file__))

FF_path = f"{CD}/1415562/ff"
Pacman_Domain = f"{CD}/domain_pacman.pddl" 
ghost_domain = f"{CD}/domain_ghost.pddl"

red_invader = 0
blue_invade = 0
red_deffender = 0
blue_defender = 0

agent1_food_eaten = 0
agent2_food_eaten = 0
total_foods = 0

agent1_closet_food = None
agent2_closet_food = None

anticipate = []


