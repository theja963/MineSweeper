import numpy as np
import pandas as pd
import random
# from functools import reduce
from scipy.ndimage import generic_filter


class mineSweeper:
    def __init__(self, position=[2,2], map_size=8, mines_ratio=1):
        self.size = map_size
        self.mines = round((mines_ratio/10)*(map_size**2))
        self.generated_map = pd.DataFrame(np.full((self.size,self.size),0))
        # self.counted_map = pd.DataFrame(np.full((self.size,self.size),0)) ##optimizing memory
        self.revealed_map = pd.DataFrame(np.full((self.size,self.size),False))
        self.flagged_map = pd.DataFrame(np.full((self.size,self.size),False)) ##optimizing memory failed
        self.win_flag = False
        self.terminate_flag = False
        self.start_position = (((int(position[0]))*self.size)+((int(position[1])))) if len(position) else self.size**2/2
        
         #incorporating first safe position
        
        
    def generate_map(self):
        random_numbers = random.sample(range(0,self.size**2),self.mines)
        for i in random_numbers:
            [col,row] = [i//self.size,i%self.size]
            self.generated_map.iat[col,row]="M"
        self.generated_map = self.count_map(self.generated_map)

    def check_window(self,values):
        surround_count = int(np.count_nonzero(values)) if values[4]!=9 else 9
        return surround_count
    
    def count_map(self,check_map):
        counted_map = generic_filter(check_map, self.check_window, size=(3,3), mode='constant', cval=0)
        # for i in counted_map:
        #     for j in i:
        #         if j==9:
        #             i="M"
        return pd.DataFrame(counted_map)

class userAction(mineSweeper):

    def generate_map(self): ## ensuring first forgiveness (may increase forgiveness values)
        self.revealed_map = pd.DataFrame(np.full((self.size,self.size),False))
        self.flagged_map = pd.DataFrame(np.full((self.size,self.size),False)) ##optimizing memory failed
        self.win_flag = False
        self.terminate_flag = False
        i = self.start_position//self.size
        j = self.start_position%self.size
        safe_zone =[[x,y] for x in
                        range(max(i-1, 0),min(i+2,self.size)) for y in
                        range(max(j-1, 0),min(j+2,self.size))]

        random_numbers = random.sample( [x for x in range(0,self.size**2) 
                                         if x != self.start_position],self.mines)
        for i in random_numbers:
            [col,row] = [i//self.size,i%self.size]
            self.generated_map.iat[col,row]=9
        self.generated_map = self.count_map(self.generated_map)
        self.reveal_map([self.start_position//self.size,self.start_position%self.size])


    def visible_map(self):
        visible_map = self.generated_map.where(self.revealed_map)
        for i in range(self.size):
            for j in range(self.size):
                if self.flagged_map.iat[i,j]:
                    visible_map.iat[i,j] = "F"
        return visible_map
        
    def reveal_map(self,position):
    #     [i,j] = position
        i,j = int(position[0]),int(position[1])
        # print ("positions", self.generated_map)
        if self.revealed_map.iat[i,j] != "F":
            
            self.revealed_map.iat[i,j] = True
            # self.revealed_map.iat[i,j] = True
            self.win_status()
            if self.generated_map.iat[i,j] in ["M",9]:
                print("Mines Triggered - Objective Failed",self.visible_map())
                self.terminate_flag = True
                return -1.0 ##scores for training
            elif self.generated_map.iat[i,j] == 0 and not self.revealed_map.iat[i,j]:
                # self.revealed_map.iat[i,j] = True
                surrounding_positions = [[x,y] for x in
                        range(max(i-1, 0),min(i+2,self.size)) for y in
                        range(max(j-1, 0),min(j+2,self.size))]
                print("Surroundings safe, unlocking more area",surrounding_positions)
                for i in surrounding_positions:
                    self.reveal_map(i)
                #reduce(self.reveal_map(x),surrounding_positions)
                return 0.05*len(surrounding_positions)
            else:
                print("Location revealed",i,j)
                return 0.01
        else:
            print("Location flagged")
            return -0.1

    def flag_map(self,position):
        [i,j] = position
        print("Flagged location",i,j)
        if not self.flagged_map.iat[i,j]:
            print("Already flagged")
            return -0.1
        else:
            if (self.flagged_map==True).sum().sum()<self.mines:    
                self.flagged_map.iat[i,j] = True
                print("Flag consumed")
                return 0.1
            elif (self.flagged_map==True).sum().sum()<self.mines:
                print("Insufficient flags")
                return -0.1
            else:
                return self.win_status()
    
    def unflag_map(self,position):
        [i,j] = position
        if self.flagged_map.iat[i,j]:
            print("Unflagged")
            self.flagged_map.iat[i,j] = False
        else:
            print("No flag placed")

    def win_status(self):
        if (self.revealed_map).sum().sum()+self.mines == self.size**2:
            print("All mines marked")
            self.win_flag = True
            self.terminate_flag=True
            return 1.0
        else:
            print("Insufficient map revealed")
            return 0.0

    ## traininic specifics required
    def to_numpy_tensor(self):
        revealed = self.revealed_map.to_numpy().astype(np.float32)
        flagged = self.flagged_map.to_numpy().astype(np.float32)
        board = self.generated_map.replace("M", 9).fillna(0).to_numpy().astype(np.float32)
        board_masked = board * revealed  # only show revealed numbers
        return np.stack([board_masked, revealed, flagged], axis=0)  # shape: (3, size, size)
    
    def training_action(self,action): ##wrapper to enable easier training
        if action =="start":
            return(self.generate_map(),self.to_numpy_tensor(),self.terminate_flag,self.win_status())
        a,x,y = action.split(",") 
        x= int(x)
        y= int(y)
        if a == "f":
            return(self.flag_map([x,y]),self.to_numpy_tensor(),self.terminate_flag,self.win_status(),self.revealed_map)
        elif a == "u":
            return(self.unflag_map([x,y]),self.to_numpy_tensor(),self.terminate_flag,self.win_status(),self.revealed_map)
        elif a == 'r':
            return(self.reveal_map([x,y]),self.to_numpy_tensor(),self.terminate_flag,self.win_status(),self.revealed_map)
        else:
            pass

    def llm_training_action(self,action):
        while not self.terminate_flag:
            response_provision = (self.stringify_visible_map,"Please choose next action as \nf:for flag, \nu:for unflag, \nr:for reveal followed by x,y coordinates\n i.e. as action,x-coordinate,y-coordinate")
            try:
                action,x,y = response_provision.split(",") 
                x= int(x)-1
                y= int(y)-1
                if action == "f":
                    return(self.flag_map([x,y]),self.terminate_flag,self.win_status(),response_provision)
                elif action == "u":
                    return(self.unflag_map([x,y]),self.terminate_flag,self.win_status(),response_provision)
                else:
                    return(self.unflag_map([x,y]),self.terminate_flag,self.win_status(),response_provision)
            except:
                print("Please enter valid input")
            return response_provision
        if execute_game_launcher.win_status() is True:
            pass
        restart = True if input(print("Would you like to restart? y/n"))not in ['n','no', 'end-game'] else False
    
    def stringify_visible_map(self):
        visible = self.visible_map().to_numpy()
        return "\\n".join([" ".join(str(cell) if pd.notna(cell) else "#" for cell in row) for row in visible])


if __name__ == "__main__":
    restart = True
    while restart == True:
        start_pos = input(print("Would you like to start new game? Please state your starting position as x,y\n"))
        start_pos = start_pos.split(",")
        print(start_pos,"values")
        execute_game_launcher = userAction(start_pos)
        execute_game_launcher.generate_map()
        while not execute_game_launcher.terminate_flag:
            response_provision = (input(print(execute_game_launcher.visible_map(),"Please choose next action as \nf:for flag, \nu:for unflag, \nr:for reveal followed by x,y coordinates\n i.e. as action,x-coordinate,y-coordinate")))
            try:
                action,x,y = response_provision.split(",") 
                x= int(x)-1
                y= int(y)-1
                if action == "f":
                    execute_game_launcher.flag_map([x,y])
                elif action == "u":
                    execute_game_launcher.unflag_map([x,y])
                else:
                    execute_game_launcher.reveal_map([x,y])
            except:
                print("Please enter valid input")
        if execute_game_launcher.win_status() is True:
            pass
        restart = True if input(print("Would you like to restart? y/n")) not in ['n','no', 'end-game'] else False



     


        
    












            


