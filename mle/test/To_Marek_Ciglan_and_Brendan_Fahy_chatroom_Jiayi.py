"""
Hi, Marek Ciglan and Brendan Fahy. My name is Jiayi. I was interviewed by you in the afternoon on Dec 09. I appreciate the time to being interviewed by you. 
It was a bit messy in the code interview, mainly due to I was nervous at the beginning and there was misunderstanding of the problem setting.
I worked on the problem on the hackerank when I had 30 mins break and made it into an API. And got the improved complexity as O(N) (N: number of posts had been seen from the stream)

Manager told me that I could send the codes to you. So I made a backup after the Hackerrank was cleared and sent it through my recruiter and schedulers.

Nice to talk to you!
"""



from collections import defaultdict
class Chat:

    def __init__(self) -> None:
        self.chatmsg = defaultdict(list) # store posts for each chatroom, sequentially, by time
        self.sender2chat = {}
        # dictionary of dictionary, store the sender send message where (chatroom) at when (timestep)
        # sender2chat: key: sender_id, val: dict of key: chatroom_id, val: list of timesteps

        self.chatdict = {} # dictionary of dictionary. key: chatroom_id, val: dict of key: timestep, val: index of msg in chatmsg[chatroom_id] # for example, if self.chatdict["42"]["81093"] == 3, this index 3 is the index of message at time "81093" in chatmsg["42"] ("42" is the chatroom id). This dict is used for given chatroom_id and timestep, look up the index of msg in chatmsg.

        self.chatlen = defaultdict(int) # key: chatroom_id, val: total number of current posts in chatroom_id

    def add(self, m: IMPost):

        if m.sender_id not in self.sender2chat:
            self.sender2chat[m.sender_id] = dict()
        if m.chatroom_id not in self.sender2chat[m.sender_id]:
            self.sender2chat[m.sender_id][m.chatroom_id] = []
        self.sender2chat[m.sender_id][m.chatroom_id].append(m.timestep)

        idx = self.chatlen(m.chatroom_id)
        if m.chatroom_id not in self.chatdict:
            self.chatdict[m.chatroom_id] = dict()
        self.chatdict[m.chatroom_id][m.timestep] = idx

        self.chatlen[m.chatroom_id] += 1
        self.chatmsg[m.chatroom_id].append(m)

    
    def find(self, chatroom_id, target_user_id, window_size):
        """
        Given chatroom_id, sender_id and window_size, determine if return the context required
        Returns:
        --------
        list of IMPost or bool: False
        """

        if window_size *2 + 1 > self.chatlen[chatroom_id]: # there is not enough context for retrievel
            return False

        # iterate ove all the timesteps in chatroom and sent by target user
        for time in self.sender2chat[target_user_id][chatroom_id]:
            # look up index of msg in chatmsg
            idx = self.chatdict[chatroom_id][time]

            # check if there is enough context for retrieval
            if idx - window_size >=0 and idx + window_size < self.chatmsg[chatroom_id]:
                return self.chatmsg[chatroom_id][idx - window_size: idx + window_size]

        return False

    
def find_conversational_context(imstream, target_user_id, window_size):

    chatApp = Chat()
    for m in imstream: # time complexity: O(N) N is the number of posts we have seen.
        chatApp.add(m)
        res = chatApp.find(m.chatroom_id, target_user_id, window_size)
        if res != False:
            return res

