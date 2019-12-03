'''BaroqueGo_BC_Player.py
The beginnings of an agent that might someday play Baroque Chess.

'''
import time
import math
import copy
import random
import BC_state_etc as BC

USE_CUSTOM_STATIC_EVAL_FUNCTION = False
USE_ZOBRIST = True
CURRENT_STATE_STATIC_VAL = 0
N_STATE_EXPANSIONS = 0
N_STATIC_EVALS = 0
N_CUTOFFS = 0
game_put_count = 0
game_get_success_count = 0
game_get_failure_count = 0
game_collision_count = 0
game_static_evals_saved_count = 0
move_put_count = 0
move_get_success_count = 0
move_get_failure_count = 0
move_collision_count = 0
move_static_evals_saved_count = 0
BOARD_SIZE = 8
WHITE = 1
BLACK = 0
PLAY_AS = WHITE
OPPO_AS = BLACK
visitedStates = {}
STOP_TIME = None
S = 64
P = 2
zobristValues = [[0] * P for i in range(S)]
remarkCnt = 0

moveDirections = ["NORTH", "EAST", "SOUTH", "WEST", "NE", "SE", "SW", "NW"]
ziggedMoveDirections = ["NORTH", "NE", "EAST", "SE", "SOUTH", "SW","WEST", "NW"]
piece_value = [0, 0, -1, 1, -2, 2, -2, 2, -2, 2, -2, 2, -100, 100, -2, 2]


# function for testing purposes
def parameterized_minimax(currentState, alphaBeta=False, ply=3,\
    useBasicStaticEval=True, useZobristHashing=False):
    '''Implement this testing function for your agent's basic
    capabilities here.'''
    global CURRENT_STATE_STATIC_VAL
    global N_STATE_EXPANSIONS
    global N_STATIC_EVALS
    global N_CUTOFFS
    global USE_CUSTOM_STATIC_EVAL_FUNCTION
    global USE_ZOBRIST
    global game_put_count
    global game_get_success_count
    global game_get_failure_count
    global game_collision_count
    global game_static_evals_saved_count
    global move_put_count
    global move_get_success_count
    global move_get_failure_count
    global move_collision_count
    global move_static_evals_saved_count

    CURRENT_STATE_STATIC_VAL = 0
    N_STATE_EXPANSIONS = 0
    N_STATIC_EVALS = 0
    N_CUTOFFS = 0

    move_put_count = 0
    move_get_success_count = 0
    move_get_failure_count = 0
    move_collision_count = 0
    move_static_evals_saved_count = 0

    if useBasicStaticEval:
        USE_CUSTOM_STATIC_EVAL_FUNCTION = False
    else:
        USE_CUSTOM_STATIC_EVAL_FUNCTION = True
    if useZobristHashing:
        USE_ZOBRIST = True
    else:
        USE_ZOBRIST = False
    # All students, add code to replace these default
    # values with correct values from your agent (either here or below).
    DATA = {}
    DATA['CURRENT_STATE_STATIC_VAL'] = -1000.0
    DATA['N_STATE_EXPANSIONS'] = 0
    DATA['N_STATIC_EVALS'] = 0
    DATA['N_CUTOFFS'] = 0
    # STUDENTS: You may create the rest of the body of this function here.

    if use_alpha_beta:
        DATA['CURRENT_STATE_STATIC_VAL'] = alphaBetaPruning(current_state,
        current_state.whose_move, max_ply, float('-inf'), float('inf'))
        DATA['N_STATE_EXPANSIONS'] = N_STATE_EXPANSIONS
        DATA['N_STATIC_EVALS'] = N_STATIC_EVALS
        DATA['N_CUTOFFS'] = N_CUTOFFS
    else:
        DATA['CURRENT_STATE_STATIC_VAL'] = minimax(current_state,
        current_state.whose_move, max_ply)
        DATA['N_STATE_EXPANSIONS'] = N_STATE_EXPANSIONS
        DATA['N_STATIC_EVALS'] = N_STATIC_EVALS
        DATA['N_CUTOFFS'] = N_CUTOFFS

    game_put_count += move_put_count
    game_get_success_count += move_get_success_count
    game_get_failure_count += move_get_failure_count
    game_collision_count += move_collision_count
    game_static_evals_saved_count += move_static_evals_saved_count

    #print(DATA)
    return DATA

# minimax algorithm
def minimax(current_state, whose_turn, max_ply, curr_ply,):
    global N_STATE_EXPANSIONS
    global N_STATIC_EVALS
    global move_put_count
    global move_get_success_count
    global move_get_failure_count
    global move_collision_count
    global move_static_evals_saved_count

    if curr_ply == 0 :
        N_STATIC_EVALS += 1
        hashKey = zhash(current_state.board)
        if USE_ZOBRIST and hashKey in visitedStates:
            item = get(hashKey)
            if item[0].board == current_state.board:
                move_get_success_count += 1
                return item[2]
            else:
                value = static_eval(current_state)
                move_collision_count += 1
                move_put_count += 1
                move_static_evals_saved_count += 1
                put(hashKey, current_state, max_ply, value, N_STATIC_EVALS)
                return value
        else:
            value = static_eval(current_state)
            put(hashKey, current_state, max_ply, value, N_STATIC_EVALS)
            move_put_count += 1
            move_get_failure_count += 1
            move_static_evals_saved_count += 1
            return value

    newStateList = exploreNextState(current_state)
    if (whose_turn == 0): # black
        N_STATE_EXPANSIONS += 1
        maxEval = float('-inf')
        for state in newStateList:
            maxEval = max(maxEval, minimax(state, 1, curr_ply-1))
        return maxEval
    elif (whose_turn == 1): # white
        N_STATE_EXPANSIONS += 1
        minEval = float('inf')
        for state in newStateList:
            minEval = min(minEval, minimax(state, 0, curr_ply-1))
        return minEval
    else:
        raise Exception("Illegal parameter for whose turn it is.")

# alpha beta pruning algorithm
def alphaBetaPruning(current_state, whose_turn, max_ply, curr_ply, alpha, beta):
    global STOP_TIME, N_STATE_EXPANSIONS, N_STATIC_EVALS, N_CUTOFFS
    global move_put_count
    global move_get_success_count
    global move_get_failure_count
    global move_collision_count
    global move_static_evals_saved_count

    if STOP_TIME != None and time.time() > STOP_TIME: return None

    if curr_ply == 0 :
        N_STATIC_EVALS += 1
        hashKey = zhash(current_state.board)
        if USE_ZOBRIST and hashKey in visitedStates:
            item = get(hashKey)
            if item[0].board == current_state.board:
                move_get_success_count += 1
                return item[2]
            else:
                value = static_eval(current_state)
                move_collision_count += 1
                move_put_count += 1
                move_static_evals_saved_count += 1
                put(hashKey, current_state, max_ply, value, N_STATIC_EVALS)
                return value
        else:
            value = static_eval(current_state)
            put(hashKey, current_state, max_ply, value, N_STATIC_EVALS)
            move_put_count += 1
            move_get_failure_count += 1
            move_static_evals_saved_count += 1
            return value

    newStateList = exploreNextState(current_state)
    if (whose_turn == 0):
        N_STATE_EXPANSIONS += 1
        maxEval = float('-inf')
        for state in newStateList:
            currVal = alphaBetaPruning(state, 1, max_ply, curr_ply-1, alpha, beta)
            if (currVal == None): return None
            if (currVal > maxEval):
                maxEval = currVal
            alpha = max(alpha, currVal)
            if (beta <= alpha):
                N_CUTOFFS += 1
                break
        return maxEval
    elif (whose_turn == 1):
        N_STATE_EXPANSIONS += 1
        minEval = float('inf')
        for state in newStateList:
            currVal = alphaBetaPruning(state, 0, max_ply, curr_ply-1, alpha, beta)
            if (currVal == None): return None
            if (currVal < minEval):
                minEval = currVal
            beta = min(beta, currVal)
            if (beta <= alpha):
                N_CUTOFFS += 1
                break
        return minEval
    else:
        raise Exception("Illegal parameter for whose turn it is.")

# Find all the possible next state from the current state
def exploreNextState(currentState):
    newStates = []
    newState = copy.deepcopy(currentState)
    totalMoveDict = getMoveListForState(newState)
    for move in totalMoveDict.keys():
        newStates.append(stateAfterMove(newState, move, totalMoveDict[move]))
    return newStates

# main function for making the move
def makeMove(currentState, currentRemark, timelimit=10):
    global PLAY_AS, OPPO_AS, STOP_TIME, remarkCnt
    global game_put_count
    global game_get_success_count
    global game_get_failure_count
    global game_collision_count
    global game_static_evals_saved_count
    global move_put_count
    global move_get_success_count
    global move_get_failure_count
    global move_collision_count
    global move_static_evals_saved_count

    move_put_count = 0
    move_get_success_count = 0
    move_get_failure_count = 0
    move_collision_count = 0
    move_static_evals_saved_count = 0
    # Compute the new state for a move.
    # You should implement an anytime algorithm based on IDDFS.
    STOP_TIME = time.time() + timelimit - 0.01;
    # The following is a placeholder that just copies the current state.
    newState = BC.BC_state(currentState.board)

    # Fix up whose turn it will be for the opponent
    newState.whose_move = 1 - currentState.whose_move
    PLAY_AS = currentState.whose_move
    OPPO_AS = 1 - PLAY_AS
    if PLAY_AS == 0:
        print("playing as: black")
    else:
        print("playing as: white")

    alpha = float('-inf')
    beta = float('inf')
    maxPlay = 1
    depthLimit = 5
    max = -999999
    min = 999999
    bestMove = None
    while (time.time() < STOP_TIME):
        moveList = getMoveListForState(currentState)
        for move in moveList.keys():
            nextStep = stateAfterMove(copy.deepcopy(currentState), move, moveList[move])
            value = alphaBetaPruning(nextStep, nextStep.whose_move, maxPlay, maxPlay, alpha, beta)
            if value is not None:
                if currentState.whose_move % 2 == PLAY_AS:
                    if value >= max:
                        max = value
                        newState = nextStep
                        bestMove = move
                else:
                    if value <= max:
                        min = value
                        newState = nextStep
                        bestMove = move
        maxPlay += 1
        if (maxPlay >= depthLimit): break

    # Make up a new remark
    utter = ["this game is gonna be easy",
             "I'm doing pretty good",
             "I might actually be able to win",
             "you just made a big mistake",
             "this is an obvious move",
             "you made a good move there",
             "this game is gonna to over soon",
             "this is too bad for you lol",
             "you are better than I thought",
             "I'm alreay tired of this",
             "I will end it soon"]
    prefix = ["Here's my move, I think ",
             "Hmmm let me think, I'd say ",
             "Hahah, do you realize that ",
             "Okay, here you go, "]

    newRemark = prefix[remarkCnt%len(prefix)] + utter[remarkCnt%len(utter)]
    remarkCnt += 1
    game_put_count += move_put_count
    game_get_success_count += move_get_success_count
    game_get_failure_count += move_get_failure_count
    game_collision_count += move_collision_count
    game_static_evals_saved_count += move_static_evals_saved_count

    print("=================================================")
    print("Report of Zobrist Hashing Status for BaroqueGO: ")
    print("PUT COUNT: " + str(game_put_count))
    print("Success GET COUNT: " + str(game_get_success_count))
    print("Failed GET COUNT: " + str(game_get_failure_count))
    print("Game Collision COUNT: " + str(game_collision_count))
    print("Saved Static Eval COUNT: " + str(game_static_evals_saved_count))
    print("Max play reached: " + str(maxPlay))
    print("=================================================")
    print("Number of States Expanded: " + str(N_STATE_EXPANSIONS))
    print("Number of Cut Offs: " +  str(N_CUTOFFS))
    print("=================================================")
    return [[bestMove, newState], newRemark]

# returns all the possible move list for the current state
def getMoveListForState(currentState):
    moveList = {}
    board = currentState.board
    oppoFreezer = None
    oppoKing = None
    myKing = None
    myPieces = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            curr = board[i][j]
            if curr == 0: continue
            if curr % 2 != currentState.whose_move:
                if curr == 14 or curr == 15:
                    oppoFreezer = (i, j)
                elif curr == 12 or curr == 13:
                    oppoKing = (i, j)
            else:
                myPieces.append((i,j))
                if curr == 12 or curr == 13:
                    myKing = (i, j)
    myPieces = immobolizeMyPieces(board, myPieces, oppoFreezer)

    for piece in myPieces:
        moveList.update(getMoveListForPiece(board, piece[0], piece[1], oppoFreezer, myKing))
    return moveList

# Returns a list of tuples contains locations that the input piece can move to
def getMoveListForPiece(board, row, col, oppoFreezer, myKing):
    moveList = {}
    currPiece = board[row][col]
    isLeaper = currPiece == 6 or currPiece == 7
    isKing = (row, col) == myKing
    isPawn = currPiece == 2 or currPiece == 3
    # Leaper special moves
    if isLeaper:
        moveList.update(getLeaperMoves(board, row, col))
    if isKing:
        moveList.update(getKingMoves(board, row, col))
    else:
        moveList.update(getNonDiagonalMoves(board, row, col, myKing))
        if not isPawn: moveList.update(getDiagonalMoves(board, row, col, myKing))
    return moveList

# returns the new state after one move
def stateAfterMove(currentState, move, captureList):
    newState = copy.deepcopy(currentState)
    moveFrom = move[0]
    moveTo = move[1]
    newState.board[moveTo[0]][moveTo[1]] = newState.board[moveFrom[0]][moveFrom[1]]
    newState.board[moveFrom[0]][moveFrom[1]] = 0
    if captureList:
        for rc in captureList:
            newState.board[rc[0]][rc[1]] = 0
    newState.whose_move = 1 - newState.whose_move
    return newState

# Initialze zobrist value
def initializeZobristValues():
    global zobristValues
    for i in range(S):
        for j in range(P):
            zobristValues[i][j] = random.randint(0, 4294967296)

initializeZobristValues()

# 0 is white piece and 1 is black piece
def zhash(board):
    global zobristValues
    val = 0;
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            piece = board[i][j]
            if piece != 0:
                if piece % 2 == 0:
                    piece = 0
                else:
                    piece = 1
                index = i*8+j
                val ^= zobristValues[index][piece]
    return val

# Get item from the hash table.
def get(hashKey):
    global visitedStates
    return visitedStates[hashKey]

# Put haskey with other items into the hash table.
def put(hashKey, state, ply_used, value, n_static_evals):
    global move_collision_count
    visitedStates[hashKey] = (state, ply_used, value, n_static_evals)

# Returns a boolean value of whether 2 input pieces are opponents
def checkOpposite(p1, p2):
    return p1 > 1 and p2 > 1 and p1 % 2 != p2 % 2

# Returns a boolean value of whether 2 input pieces are allies
def checkAlly(p1, p2):
    return p1 > 1 and p2 > 1 and p1 % 2 == p2 % 2

# Name of this agent
def nickname():
    return "BaroqueGo"

# Introduce our agent
def introduce():
    return "I'm Baroque Go, created by Erik(huangti) and Mark(lmh98), let's play a round of Baroque chess!"

# Prepare to start
def prepare(player2Nickname):
    return player2Nickname + ", you better be prepared to lose, I'm really aggressive"

# Mian static evaluation function
def static_eval(state):
    if USE_CUSTOM_STATIC_EVAL_FUNCTION:
        return customStaticEval(state)
    else:
        return basicStaticEval(state)

# Custom static evaluation function
def customStaticEval(state):
    board = state.board
    value = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            currPiece = board[i][j]
            if currPiece % 2 == state.whose_move: continue
            for direction in moveDirections:
                for rc in getIndicesInDirection(i,j,1,direction):
                    if board[rc[0]][rc[1]] == 0:
                        value += 5
                    elif checkOpposite(currPiece, board[rc[0]][rc[1]]):
                        if currPiece == 14 or currPiece == 15:
                            value += 20
                        value += abs(piece_value[board[rc[0]][rc[1]]])
    return value

# Basic static evaluation function
def basicStaticEval(state):
    board = state.board
    value = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            currPiece = board[i][j]
            if currPiece % 2 == state.whose_move:
                value += abs(piece_value[board[i][j]])
            else:
                value -= abs(piece_value[board[i][j]])
    return value

# Find the array of my immobolized pieces and output it
def immobolizeMyPieces(board, myPieces, oppoFreezer):
    if oppoFreezer == None:
        return myPieces
    row = oppoFreezer[0]
    col = oppoFreezer[1]
    result = myPieces
    for direction in moveDirections:
        for rc in getIndicesInDirection(row,col,1,direction):
            if checkOpposite(board[row][col], board[rc[0]][rc[1]]):
                result.remove((rc[0], rc[1]))
    return result

# special move list generation for king
def getKingMoves(board, row, col):
    moveList = {}
    currPiece = board[row][col]
    for direction in moveDirections:
        for rc in getIndicesInDirection(row,col,1,direction):
            if board[rc[0]][rc[1]] == 0:
                moveList[((row, col), (rc[0], rc[1]))] = []
            elif checkOpposite(currPiece, board[rc[0]][rc[1]]):
                moveList[((row, col), (rc[0], rc[1]))] = [(rc[0], rc[1])]
    return moveList

# Check non diaoonal directions
def getNonDiagonalMoves(board, row, col, myKing):
    moveList = {}
    for direction in moveDirections[:4]:
        for rc in getIndicesInDirection(row,col,BOARD_SIZE,direction):
            if board[rc[0]][rc[1]] == 0:
                captured = getCaptured(board, row, col, rc[0], rc[1], direction, myKing)
                moveList[((row, col), (rc[0], rc[1]))] = captured
            else:
                break
    return moveList

# Check diagonal directions
def getDiagonalMoves(board, row, col, myKing):
    moveList = {}
    for direction in moveDirections[4:]:
        for rc in getIndicesInDirection(row,col,BOARD_SIZE,direction):
            if board[rc[0]][rc[1]] == 0:
                captured = getCaptured(board, row, col, rc[0], rc[1], direction, myKing)
                moveList[((row, col), (rc[0], rc[1]))] = captured
            else:
                break
    return moveList

# get special moves for leaper piece
def getLeaperMoves(board, row, col):
    moveList = {}
    disp = 1
    currLeaper = board[row][col]
    for direction in moveDirections:
        slots = getIndicesInDirection(row,col,BOARD_SIZE,direction)
        for i in range(len(slots)-1):
            oppoLoc = slots[i]
            leapLoc = slots[i+1]
            canLeap = board[leapLoc[0]][leapLoc[1]] == 0
            hasOpponent = checkOpposite(currLeaper, board[oppoLoc[0]][oppoLoc[1]])
            if canLeap and hasOpponent:
                moveList[((row, col), leapLoc)] = [oppoLoc]
            else:
                break
    return moveList

# get array of indices in a certain direction
def getIndicesInDirection(row, col, disp, direction):
    indices = []
    if direction == "NORTH":
        for y in range(1, disp+1):
            if row-y >= 0:
                indices.append((row-y,col))
        return indices
    if direction == "SOUTH":
        for y in range(1, disp+1):
            if row+y < BOARD_SIZE:
             indices.append((row+y,col))
        return indices
    if direction == "WEST":
        for x in range(1, disp+1):
            if col-x >= 0:
                indices.append((row,col-x))
        return indices
    if direction == "EAST":
        for x in range(1, disp+1):
            if col+x < BOARD_SIZE:
                indices.append((row,col+x))
        return indices
    if direction == "NW":
        for k in range(1, disp+1):
            if row-k >= 0 and col-k >= 0:
                indices.append((row-k,col-k))
        return indices
    if direction == "NE":
        for k in range(1, disp+1):
            if row-k >= 0 and col+k < BOARD_SIZE:
                indices.append((row-k,col+k))
        return indices
    if direction == "SW":
        for k in range(1, disp+1):
            if row+k < BOARD_SIZE and col-k >= 0:
                indices.append((row+k, col-k))
        return indices
    if direction == "SE":
        for k in range(1, disp+1):
            if row+k < BOARD_SIZE and col+k < BOARD_SIZE:
                indices.append((row+k, col+k))
        return indices

# get an array of captured pieces from a given move
def getCaptured(board, r1, c1, r2, c2, direction, myKing):
    currPiece = board[r1][c1]
    # LEAPER AND FREEZER DONT KILL IN DIAG & NONDIAG MOVES
    if currPiece == 14 or currPiece == 15 or currPiece == 6 or currPiece == 7:
        return []
    # Withdrawer CASE
    if currPiece == 10 or currPiece == 11:
        killDirection = ziggedMoveDirections[(ziggedMoveDirections.index(direction) + 4) % 8]
        rc = getIndicesInDirection(r1, c1, 1, killDirection)
        if rc and checkOpposite(currPiece, board[rc[0][0]][rc[0][1]]):
            return rc
        else:
            return []
    # PAWN CASE
    if currPiece == 2 or currPiece == 3:
        killList = []
        for direction in moveDirections[:4]:
            enemyLocs = getIndicesInDirection(r2,c2,1,direction)
            if enemyLocs: # has a piece
                rc1 = enemyLocs[0] # location tuple
                if checkOpposite(currPiece, board[rc1[0]][rc1[1]]): # if this is an enemy
                    allyLocs = getIndicesInDirection(rc1[0],rc1[1],1,direction)
                    if allyLocs: # has a piece
                        rc2 = allyLocs[0] # location tuple
                        if checkAlly(currPiece, board[rc2[0]][rc2[1]]): # if this is an ally
                            killList.append(rc1)
        return killList
    # COORDINATOR CASE
    if currPiece == 4 or currPiece == 5:
        killList = []
        if checkOpposite(currPiece, board[myKing[0]][c2]):
            killList.append((myKing[0], c2))
        if checkOpposite(currPiece, board[r2][myKing[1]]):
            killList.append((r2, myKing[1]))
        return killList
