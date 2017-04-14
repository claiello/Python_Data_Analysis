import numpy as np

def get_data(fname):
    
    content = []
    
    cnt = 0
    
    tr = 0
    
    next_tr = False
    
    hlp = []
    with open(fname) as f:
        for line in f:
           cnt += 1
           if cnt >= 8 and next_tr == False:
                my_line = line.strip()
                my_line = my_line.split('\t')
                #print my_line
                if len(my_line) > 2:
                    for k in range(len(my_line)-1):
                        my_line[k] = np.float(my_line[k])
                hlp.append(my_line)
                if line[0:3] == '---':
                    next_tr = True
           else:           
               if next_tr:
                   next_tr = False
                   content.append(np.array(hlp))
                   hlp = []
                   cnt = 0
    
           if 'str' in line:
              break
    
    
    # you may also want to remove whitespace characters like `\n` at the end of each line
    
    all_particles = []
    for k in range(len(content)):
        new_content = np.array([])
        hlp = content[k]
        hlp = hlp[:-1]
        hlp2 = np.zeros([len(hlp), 7])
        for m in range(len(hlp)):
            hlp2[m, :] = np.array(hlp[m][:-1])
    
        new_content = np.vstack((new_content, hlp2)) if new_content.size else hlp2
    
        all_particles.append(new_content)
    
    #print all_particles

    print("Found " + str(len(all_particles)) + " trajectories.")

    return all_particles



