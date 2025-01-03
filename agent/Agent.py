import numpy as np
import copy

GRID_WIDTH = 10
GRID_DEPTH = 20

BLOCK_LENGTH = 4
BLOCK_WIDTH = 4

ipieces = [[[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
          [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
          [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
          [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]]
opieces = [[[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]]]
jpieces = [[[0, 3, 3, 0], [0, 0, 3, 0], [0, 0, 3, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 3, 3, 3], [0, 3, 0, 0], [0, 0, 0, 0]],
          [[0, 0, 3, 0], [0, 0, 3, 0], [0, 0, 3, 3], [0, 0, 0, 0]],
          [[0, 0, 0, 3], [0, 3, 3, 3], [0, 0, 0, 0], [0, 0, 0, 0]]]
lpieces = [[[0, 0, 4, 0], [0, 0, 4, 0], [0, 4, 4, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 4, 4, 4], [0, 0, 0, 4], [0, 0, 0, 0]],
          [[0, 0, 4, 4], [0, 0, 4, 0], [0, 0, 4, 0], [0, 0, 0, 0]],
          [[0, 4, 0, 0], [0, 4, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0]]]
zpieces = [[[0, 5, 0, 0], [0, 5, 5, 0], [0, 0, 5, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 5, 5, 0], [5, 5, 0, 0], [0, 0, 0, 0]],
          [[0, 5, 0, 0], [0, 5, 5, 0], [0, 0, 5, 0], [0, 0, 0, 0]],
          [[0, 0, 5, 5], [0, 5, 5, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
spieces = [[[0, 0, 6, 0], [0, 6, 6, 0], [0, 6, 0, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 6, 6, 0], [0, 0, 6, 6], [0, 0, 0, 0]],
          [[0, 0, 6, 0], [0, 6, 6, 0], [0, 6, 0, 0], [0, 0, 0, 0]],
          [[6, 6, 0, 0], [0, 6, 6, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
tpieces = [[[0, 0, 7, 0], [0, 7, 7, 0], [0, 0, 7, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 7, 7, 7], [0, 0, 7, 0], [0, 0, 0, 0]],
          [[0, 0, 7, 0], [0, 0, 7, 7], [0, 0, 7, 0], [0, 0, 0, 0]],
          [[0, 0, 7, 0], [0, 7, 7, 7], [0, 0, 0, 0], [0, 0, 0, 0]]]
lspieces = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8] #this is the lines sent piece aka garbage lines

PIECES_DICT = {
    'I': ipieces, 'O': opieces, 'J': jpieces,
    'L': lpieces, 'Z': zpieces, 'S': spieces,
    'T': tpieces, 'G': lspieces
}

PIECE_NUM2TYPE = {1: 'I', 2: 'O', 3: 'J', 4: 'L', 5: 'Z', 6: 'S', 7: 'T', 8: 'G'}
PIECE_TYPE2NUM = {val: key for key, val in PIECE_NUM2TYPE.items()}
POSSIBLE_KEYS = ['I', 'O', 'J', 'L', 'Z', 'S', 'T']

def heightForColumn(grid, column):
    height, width = grid.shape
    for i in range(0, height):
        if grid[i][column] != 0:
            return height-i
    return 0

def heights(grid):
    result = []
    height, width = grid.shape
    for i in range(0, width):
        result.append(heightForColumn(grid, i))
    return result

def numberOfHoleInColumn(grid, column):
    result = 0
    maxHeight = heightForColumn(column)
    for height, line in enumerate(reversed(grid)):
        if height > maxHeight: break
        if line[column] == 0 and height < maxHeight:
            result+=1
    return result

def numberOfHoleInRow(grid, line):
    result = 0
    height, width = grid.shape
    for index, value in enumerate(grid[height-1-line]):
        if value == 0 and heightForColumn(index) > line:
            result += 1
    return result

def lines_cleared_potential(grid, current_piece):
    """
    Tính toán số dòng tiềm năng có thể xóa được bằng khối hiện tại.
    """
    best_lines_cleared = 0

    #temp_grid = grid.copy() #bạn không thay đổi grid nên không cần dùng deepcopy
    for rotation in range(len(current_piece)):
        for offset in range(-3, GRID_WIDTH):  # Thử tất cả offset
            temp_grid = copy.deepcopy(grid) #sửa lại, đặt temp_grid vào trong vòng lặp for
            py = -2
            while not collide(temp_grid.T, current_piece[rotation], offset, py):
              py += 1

            # Đặt khối
            for x in range(BLOCK_WIDTH):
                for y in range(BLOCK_LENGTH):
                    if current_piece[rotation][y][x] > 0:
                        if 0 <= offset + x < GRID_WIDTH and 0 <= py + y < GRID_DEPTH:
                           temp_grid[GRID_DEPTH - 1 - (py + y)][offset + x] = current_piece[rotation][y][x]

            lines_cleared = completLine(temp_grid)
            best_lines_cleared = max(best_lines_cleared, lines_cleared)

    return best_lines_cleared

def heuristic_hold_I_for_tetris(grid, hold_piece, next_piece):
    """
    Heuristic khuyến khích hold khối I.
    Thưởng điểm nhiều hơn nếu có thể ăn 2, 3, hoặc 4 hàng sau khi hold I.
    """
    hold_piece_id = 0
    if hold_piece != 0:
        for row in hold_piece[0]:
            for cell in row:
                if cell != 0:
                    hold_piece_id = cell
                    break
            if hold_piece_id != 0:
                break


    if PIECE_NUM2TYPE.get(hold_piece_id) == 'I':
        lines_cleared = lines_cleared_potential(grid, hold_piece)
        if lines_cleared >= 4:  # Tetris
          return lines_cleared**2  # Thưởng rất nhiều điểm
        elif lines_cleared >= 2: # double, triple
           return lines_cleared*3   # Thưởng nhiều điểm
        
        else: 
            return 1


    elif next_piece!=0 and next_piece[0][0][0] == 1:

        #Nếu next_piece là khối I
       return 1

    
    else:#nếu không phải khối I và hold không phải I hoặc hold rỗng
        return 0

def combined_height_holes(heights, num_holes): #heuristic tổng hợp chiều cao cột và số lỗ
    """Kết hợp chiều cao và số lỗ."""
    return sum(heights) * (1+ sum(num_holes))

def flatten_list(nested_list):
    """Làm phẳng một list phức tạp thành một list 1 chiều chứa các số."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # Gọi đệ quy nếu gặp list
        elif isinstance(item, (int, float, np.integer, np.floating)): #nếu item là số thì append vào
            flat_list.append(item)
        elif isinstance(item, np.ndarray): #nếu là mảng numpy thì convert sang list và extend vào
            flat_list.extend(item.flatten().tolist())
    return flat_list
    
def heuristics(grid, hold_piece, next_piece):
    columns_heights = heights(grid)
    num_holes_ = numberOfHoles(grid, columns_heights)  # Lưu số lỗ vào biến
    ho_penalty = max((max(columns_heights) - 17) / 2 ,0) #chỉ phạt nếu chiều cao quá lớn
    max_height = max(max(columns_heights) - 12, 0)
    
    return flatten_list(columns_heights + [aggregateHeight(columns_heights)] + numberOfHoles(grid, columns_heights) + bumpinesses(columns_heights) + [completLine(grid), maxPitDepth(columns_heights), maxHeightColumns(columns_heights), minHeightColumns(columns_heights), heuristic_hold_I_for_tetris(grid, hold_piece, next_piece), ho_penalty, max_height])



def aggregateHeight(heights):
    result = sum(heights)
    return result

def bumpinesses_2(heights):
    result = []
    id = -1 
    min = 30
    for i in range(len(heights)):
        if heights[i] < min:
            min = heights[i]
            id = i
        
    for i in range(0, len(heights)-1):
        if i == id:
            continue
        result.append(abs(heights[i]-heights[i+1]))
    return result

def aggregatebumpinesses(heights):
    bum = bumpinesses_2(heights)
    result = sum(bum)
    return result

def completLine(grid):
    result = 0
    height, width = grid.shape
    for i in range (0, height) :
        if 0 not in grid[i]:
            result+=1
    return result

def bumpinesses(heights):
    result = []
    for i in range(0, len(heights)-1):
        result.append(abs(heights[i]-heights[i+1]))
    return result

def numberOfHoles(grid, heights):
    results = []
    height, width = grid.shape
    for j in range(0, width) :
        result = 0
        for i in range (0, height) :
            if grid[i][j] == 0 and height-i < heights[j]:
                result+=1
        results.append(result)
    return results

def maxHeightColumns(heights):
    return max(heights)

def minHeightColumns(heights):
    return min(heights)

def maxPitDepth(heights):
    return max(heights)-min(heights)

def get_feasible(block):
    feasibles = []

    b = block

    for x in range(BLOCK_WIDTH):
        for y in range(BLOCK_LENGTH):
            if b[x][y] > 0:
                feasibles.append([x, y])

    return feasibles

def collide(grid, block, px, py):
    feasibles = get_feasible(block)

    for pos in feasibles:
        # print(px + pos[0], py + pos[1])
        if px + pos[0] > GRID_WIDTH - 1:   # right
            return True

        if px + pos[0] < 0:   # left
            return True

        if py + pos[1] > len(grid[0]) - 1:  # down
            return True

        if py + pos[1] < 0:   # up
            continue

        if grid[px + pos[0]][py + pos[1]] > 0:
            # print(px, py)
            # print(px + pos[0], py + pos[1])
            # print("Touch")
            return True

    return False

def collideDown(grid, block, px, py):
    return collide(grid, block, px, py + 1)

def hardDrop(grid, block, px, py):
    y = 0
    x = 0
    if collideDown(grid, block, px, py) == False:
        x = 1
    if x == 1:
        while True:
            py += 1
            y += 1
            if collideDown(grid, block, px, py) == True:
                break

    return y

def get_block_list_id(one_hot):
  for i in range(len(one_hot)):
    if one_hot[i] == 1:
      return i + 1
  return 0

def get_grid(px, block, state):
    return_grids = np.zeros(shape=(10, 20), dtype=np.float32)

    grid = copy.deepcopy(state[:, :10].reshape(20, 10).transpose())
    grid[grid == 0.7] = 0.0
    grid[grid == 0.3] = 0.0

    block, px, py = block, px, -2

    add_y = hardDrop(grid, block, px, py)

    for x in range(0,4):
        for y in range(0,4):
            if block[x][y] > 0:
                # draw ghost grid
                if -1 < px + x < 10 and -1 < py + y + add_y < 20:
                    grid[px + x][py + y + add_y] = 0.3

                if -1 < px + x < 10 and -1 < py + y < 20:
                    grid[px + x][py + y] = 0.7
    return grid.transpose()

def cnt_cleared_lines(state):
    state = state[:, :10].reshape(20, 10)
    state[state == 0.3] = 1.0
    state[state != 1.0] = 0.0
    cnt_full_lines = 0
    for line in state:
      check = True
      for cell in line:
        if cell == 0:
          check = False
          break
      if check == True:
        cnt_full_lines += 1

    return cnt_full_lines

def lines_for_combo(n_combos):
    if n_combos == -1:
      return -1
    elif n_combos == 0:
      return 0
    elif n_combos == 1 or n_combos == 2:
      return 1
    elif n_combos == 3 or n_combos == 4:
      return 2
    elif n_combos == 5 or n_combos == 6:
      return 3
    else:
      return 4


class Agent:
    def __init__(self, turn):
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # weight_file_path = os.path.join(dir_path, turn, 'weight')

        self.moves = []
        self.weights=[0.3348691800975014, -0.5115113093701886, 0.10216463360538339, -0.9476099463931876, -0.4607235940225171, -0.23562661578393862, 0.4594408305881734, -0.7518293657433541, -1.0699410819646176, 0.579526455967861, -3.7355573017762214, -1.558544939373595, -0.7685312120233372, -0.8319268939816978, -0.6174758485197249, 0.5147881199079448, 0.06343189249317058, -0.5862157906887837, -1.233912936135763, -0.356657891473697, 0.260530957330605, -1.1272757838469794, -0.20439811524633428, -0.8542608322803131, -0.14314887149091485, -0.14735523772770595, 0.43343972281532955, -1.5676094574416979, -1.1617280255491886, -0.8325122390933486, 0.6311337877510816, -0.36900453901342894, -2.154233984876191, 0.27501126030564366, 0.40066001100104487, -0.6831028977829721, 1.1857873525802405, 0.5271237911211138, -0.8413194258216087, -0.930490167896317, -0.9394556842617436, 0.8108388606408721, 2.8457255103520147, 0.6068774956309034, 0.5632461436919777, -0.32748377304354903, -0.5771400255329507, 0.8185084680395736, 0.30605707647702995, 0.6581555845901361, -1.7560638175161323, -1.1371745289005621, -0.28410865950140607, 1.5252126012125, 2.2903726823646195, -0.6020372062767658, -0.8798079688445449, 0.3129089443141543, 1.7624296690730807, 3.922973773178526, -0.6721354491121379, -0.7214678616265633, 0.9682681365085741, -1.1703934737970283, 1.0441491024496523, 0.8546707780603813, 0.6600529188018972, 1.1475189052762287, 0.07594007428751384, -0.5869196285299518, 0.6118627141957748, 1.5650198761244285, 0.6132408883621722, -0.21815891782737712, -0.2709054964481553, -0.6368751995467753, -2.0620946874407213, -0.7260605759881209, -2.9080499392399073, 1.2574057142406285, -0.15794598382476005, 0.9211099612472271, 0.8971089235813232, -1.8509528395178567, -1.9576481153926895, 0.5, 0.5]
        
    def best_action(self, state):
        n_block_list_id = get_block_list_id(state[6, 10:17].reshape(7))
        h_block_list_id = get_block_list_id(state[0, 10:17].reshape(7))
        f_block_list_id = get_block_list_id(state[1, 10:17].reshape(7))

        bestRotation = None
        bestOffset = None
        bestScore = None
        bestSwap = None

        possible_blocks = []

        if h_block_list_id == 0:
          hold_piece = 0 #hold rỗng
          possible_blocks = [PIECES_DICT[PIECE_NUM2TYPE[n_block_list_id]], PIECES_DICT[PIECE_NUM2TYPE[f_block_list_id]]]
        else:
          hold_piece = PIECES_DICT[PIECE_NUM2TYPE[h_block_list_id]] # Lấy khối đang hold
          possible_blocks = [PIECES_DICT[PIECE_NUM2TYPE[n_block_list_id]], PIECES_DICT[PIECE_NUM2TYPE[h_block_list_id]]]

        bestScore_ = [0, 0]
        bestOffset_ = [0, 0]
        bestRotation_ = [0, 0]

        my_combo = copy.deepcopy(state[7, 11]).reshape(1)*10
        block_infos = copy.deepcopy(state[0:7, 10:17]).reshape(7,7).flatten()
        lines_about_to_get = copy.deepcopy(state[7, 13]).reshape(1)*20

        for i in range(len(possible_blocks)):

            if i == 0:  # Nếu không hold
                next_piece = PIECES_DICT[PIECE_NUM2TYPE[f_block_list_id]] #Khối tiếp theo sẽ là khối f
                if h_block_list_id == 0: # Nếu hố hold trống, thì hold_piece=0
                    hold_piece = 0  # Hold rỗng
                else: # Nếu không thì hold_piece là khối hiện tại đang được giữ
                    hold_piece = PIECES_DICT[PIECE_NUM2TYPE[h_block_list_id]]
            else:  # Nếu hold
                next_piece = PIECES_DICT[PIECE_NUM2TYPE[n_block_list_id]] #khối tiếp theo sẽ là khối hiện tại
                hold_piece = PIECES_DICT[PIECE_NUM2TYPE[n_block_list_id]]  # hold_piece là khối hiện tại
            
            for rotation in range(0, 4):
                cnt_check = np.count_nonzero(np.array(possible_blocks[i][rotation]).flatten() != 0)

                for offset in range(-3, 11):
                    changed_board = get_grid(offset, possible_blocks[i][rotation], state)
    
                    if np.count_nonzero(np.array(changed_board).flatten() == 0.3) == cnt_check:

                        changed_board[changed_board == 0.3] = 1.0
                        changed_board[changed_board != 1.0] = 0.0
    
                        heuristic = np.concatenate((heuristics(changed_board, hold_piece, next_piece),my_combo,lines_about_to_get,block_infos))
    
                        score = sum([a*b for a,b in zip(heuristic, self.weights)])
                        # Tính số lỗ hiện tại
                        heights_current = heights(changed_board)
                        holes_current = sum(numberOfHoles(changed_board, heights_current))
                        # Trừ điểm nếu số lỗ tăng lên
                        if holes_current>0:
                            score-=holes_current*700

                        if possible_blocks[i][rotation] == PIECES_DICT['I']:
                            if cnt_cleared_lines(state)>=3:
                                score += 15000  # Cộng điểm cho việc đặt chữ "I"
                            else:
                                score -=5000

                        columns_heights = heights(changed_board)

                        max_height = max(max(columns_heights) - 12, 0)
                        score -= max_height*200
                        ab = aggregatebumpinesses(columns_heights)
                        score -= ab*300
                        
                        if bestScore_[i] == 0 or score > bestScore_[i]:
                            bestScore_[i] = score
                            bestOffset_[i] = offset
                            bestRotation_[i] = rotation


        if bestScore_[0] > bestScore_[1]:
          bestRotation = bestRotation_[0]
          bestOffset = bestOffset_[0]
          bestScore = bestScore_[0]
          bestSwap = 0
        else:
          bestRotation = bestRotation_[1]
          bestOffset = bestOffset_[1]
          bestScore = bestScore_[1]
          bestSwap = 1

        return bestOffset, bestRotation, bestSwap, bestScore

    def choose_action(self, state):
        if(len(self.moves) > 0):
          move =self.moves[0]
          self.moves.pop(0)
          return move

        offset, rotation, swap, _ = Agent.best_action(self, state)

        if swap == 1:
          self.moves.append(1)

        if offset != None:
          offset = offset - 4
          if rotation == 3:
            self.moves.append(4)
          else:
            for _ in range(0, rotation):
                self.moves.append(3)

          for _ in range(0, abs(offset)):
              if offset > 0:
                  self.moves.append(5)
              else:
                  self.moves.append(6)

          self.moves.append(2)

        move =self.moves[0]
        self.moves.pop(0)
        return move