from arm_dynamics_base import ArmDynamicsBase
import numpy as np
from geometry import rot, xaxis, yaxis


class ArmDynamicsStudent(ArmDynamicsBase):


    def find_rot_matrix_w(link,state):
        
        i = 0
        qtot = 0
        while i < link:
            qtot += state[i]
            i+=1
        Rmat = np.transpose(np.array[[np.cos(qtot), -np.sin(qtot)], [np.sin(qtot), np.cos(qtot)]])
        return Rmat
    
    
    def find_rot_matrix_next(qnext):
        
        Rmat = np.transpose(np.array[[np.cos(qnext), -np.sin(qnext)], [np.sin(qnext), np.cos(qnext)]])
        return Rmat

    def dynamics_step(self, state, action, dt):
        # state has the following format: [q_0, ..., q_(n-1), qdot_0, ..., qdot_(n-1)] where n is the number of links
        # action has the following format: [mu_0, ..., mu_(n-1)]
        # You can make use of the additional variables:
        # self.num_links: the number of links
        # self.joint_viscous_friction: the coefficient of viscous friction
        # self.link_lengths: an array containing the l of all the links
        # self.link_masses: an array containing the m of all the links
        
        time_step = dt
        link_num = self.num_links
        phi = self.joint_viscous_friction
        l = self.link_lengths[0]
        m = self.link_masses[0]
        #LHS_matrix = np.empty((link_num*10,link_num*10))
        if link_num == 1:
        
           
        
            Row1 = np.array([1,0,0,0,0,-m,0,0,0,0])
            Row2 = np.array([0,1,0,0,0,0,-m,0,0,0])
            Row3 = np.array([0,0,0,-1,0,1,0,0,0,0])
            Row4 = np.array([0,0,0,0,-1,0,1,0,-0.5*l,0])
            Row5 = np.array([0,-0.5*l,1,0,0,0,0,0,-m*l*l*(1/12),0])
            Row6 = np.array([0,0,0,1,0,0,0,0,0,0])
            Row7 = np.array([0,0,0,0,1,0,0,0,0,0])
            Row8 = np.array([0,0,1,0,0,0,0,0,0,0])
            Row9 = np.array([0,0,0,0,0,0,0,0,-1,1])
            Row10 = np.array([0,0,0,0,0,0,0,1,0,0])
            
            LHS = np.array([Row1, Row2, Row3, Row4, Row5, Row6, Row7, Row8, Row9, Row10])
            RHS = np.array([-m*9.81*np.sin(state[0][0]), m*9.81*np.cos(state[0][0]),-0.5*m*l*state[1][0]*state[1][0],0, phi*state[1][0],0,0,action[0][0], 0,state[0][0]])
            
            
            sol = np.linalg.solve(LHS,RHS)
          
            qddot = sol[len(sol)-1]
            
            
            newqd = state[1][0] + qddot*time_step
            newq = state[0][0] + state[1][0]*time_step + 0.5*qddot*time_step**2
            
            finalq = [[newq], [newqd]]
            state = finalq
        if link_num == 2:
           #print(state)
           l2 = self.link_lengths[1]
           m2 = self.link_masses[1]
           q1 = state[0][0]
           q2 = state[1][0]
           qd1 = state[2][0]
           qd2 = state[3][0]
           w1 = qd1
           w2 = qd2
           mu1 = action[0][0]
           mu2 = action[1][0]
           
           Row1 = np.array([1,0,-np.cos(q2),np.sin(q2),-m,0,0,0,0,0,0,0])
           Row2 = np.array([0,1,-np.sin(q2),-np.cos(q2),0,-m,0,0,-m*l*0.5,0,0,0])
           Row3 = np.array([0,-0.5*l,-0.5*l*np.sin(q2),-0.5*l*(np.cos(q2)),0,0,0,0,-m*l*l/12,0,0,0])
           Row4 = np.array([0,0,0,0,1,0,0,0,0,0,0,0])
           Row5 = np.array([0,0,0,0,0,1,0,0,0,0,0,0])
           Row6 = np.array([0,0,0,0,0,0,0,0,-1,1,0,0])
           Row7 = np.array([0,0,1,0,0,0,-m2,0,0,0,0,0])
           Row8 = np.array([0,0,0,1,0,0,0,-m2,0,0,-m2*l2*0.5,0])
           Row9 = np.array([0,0,0,-0.5*l2,0,0,0,0,0,0,-m2*l2*l2/12,0])
           Row10 = np.array([0,0,0,0,np.cos(-q2),np.sin(-q2),-1,0,l*np.sin(-q2),0,0,0])
           Row11 = np.array([0,0,0,0,-np.sin(-q2),np.cos(-q2),0,-1,l*np.cos(-q2),0,0,0])
           Row12 = np.array([0,0,0,0,0,0,0,0,1,0,-1,1])
           
           
           LHS = np.array([Row1,Row2,Row3,Row4,Row5,Row6,Row7,Row8,Row9,Row10,Row11,Row12])
           
           RHS = np.array([-m*9.81*np.sin(q1)-0.5*m*l*w1*w1, m*9.81*np.cos(q1), phi*qd1 - mu1 +mu2, 0,0,0, 
           -m2*9.81*np.sin(q1+q2) - 0.5*m2*l2*w2*w2, m2*9.81*np.cos(q1+q2), phi*qd2 - mu2, np.cos(-q2)*l*w1*w1, 
           -np.sin(-q2)*l*w1*w1, 0])
           
           sol = np.linalg.solve(LHS,RHS)
          
           qddot2 = sol[len(sol)-1]
           qddot1 = sol[len(sol) - 3]
            
            
           newqd1 = qd1 + qddot1*time_step
           newq1 = q1 + qd1*time_step + 0.5*qddot1*time_step**2
           newqd2 = qd2 + qddot2*time_step
           newq2 = q2 + qd2*time_step + 0.5*qddot2*time_step**2
            
           finalq = [[newq1], [newq2], [newqd1], [newqd2]]
           state = finalq
           	
           #print(finalq)
           #print(w1)
           #print(w2)
        
        # Replace this with your code:
        return state
