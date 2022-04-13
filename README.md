# pacman-contest
Pacman AI contest for COSC-241

Our Implementation for Pacman Capture The Flag taught in COSC-241: Artificial Intelligence

Team: Ahmed Aly, Brandon Ngacho, Daniel Flores Garcia

## Introduction:
Coming up with a plan for CTF was the task that took most of our time in the tournament. We had to come up with lots of different plans and evaluate each of them to know what to start implementing. We, initially, wanted to have two balanced agents that are both attacking and defending. However, we thought that this would be both time consuming, not yield a strong enough team, and would be harder to divide tasks through and progress as a team. Thus, we decided to divide our time and team into two agents: offensive and defensive. We then brainstormed the strategies to come up with for both agents. We decided on approaching these strategies by creating very simple agents and improving them over time. The following sums up the details of our agents.

## Offense:
### Introduction:
Creating an offensive agent proved to be a very tricky task. In comparison with the defensive agent where our mission was pretty clear, the offensive agent doesn’t have a really direct way to his goal. How can we make an agent that accumulates food, evades ghosts, successfully traverses the maze, possibly eats ghosts, deposits food back, and repeats all while we have to meet a strict, looming deadline? We decided to create a crude agent and keep improving it around the clock. We initially began with a simple food greedy agent. We then needed to make our agent take ghosts into consideration and create a state evaluation function. This task was more complicated than it appeared on the surface. Ghost behavior varies largely between different teams, and not all ghosts are equal; one might be attacking and one defending. Furthermore, ghosts could be scared of you or not. Then there is also the fact that they could be just lying across the borders as Pacman, which might result in them attacking you quickly once you try to deposit food. 

We managed to get this function to work and almost win against the baseline team all the time, never losing. However, this wasn’t enough for our team as we vowed to keep improving our agent and there were lots of flaws. Our biggest problem was that our code didn’t seem to utilize a lot of the concepts we learned in class. Essentially, it was a massive evaluation function consisting of mostly nested if-else statements and arbitrary set constants. Thus, we agreed to implement a method of reinforcement learning as we had a good adversary to train against, multiple different maze layouts to traverse, and, most importantly, a computer that can keep looping for multiple episodes. We chose Approximate TD Q Learning as it was the mode of learning we were most familiar with, had written code for before, and seemed the most efficient out of the modes of learning we had learned.

### Implementation:
It was very tricky to get learning to work as we needed to carefully set our feature functions, reward functions, hold training episodes, and ensure that our learning code actually works.
In the end, we came up with the following feature and reward functions:
#### Features:
‘#-of-ghosts-1-step-away’ is a discrete feature that returns how many enemy ghosts are one step away in the successor state.
‘enemy1scared’ is a binary feature that is toggled as long as the 1st enemy is scared as a result of us eating a power pellet. This feature then takes away the fear our agent had regarding the enemies (it assumes the other enemy’s effect is negligble). This led to us having to shuffle the baselineTeam agents while training, so our model isn’t generalizable enough. It’d then toggle another continuous feature: 'distanceToScared1Ghost'.
‘distanceToScared1Ghost’ is a continuous feature that calculates the distance to that scared ghost. We would calculate the distances through running a particle filtering algorithm then take the most probable position as the ghost’s position. A major problem with this is that the ghost isn’t necessarily a big danger if it’s far enough and our particle filtering algorithm has a high built-in random error. These might lead to our agent getting confused and possibly not be able to associate the distance with something major
‘eats-ghost1’ is a binary feature that would be on only if ‘enemy1scared’ is on, and ghost1 is in one of our successor positions. Of course, this might lead to time waste if the enemy is trying to waste our time. To minimize that effect and still manage to eat the ghost if possible, we wouldn’t toggle that feature if the ghost was a move away from not being scared. Furthermore, since we only eat the ghost if it’s in one of our successor positions, we don’t actively hunt it, so if it dances around like our defensive agent does, we simply wouldn’t care and keep on cumulating food
All of these features also exist for the second ghost.
‘eatsFood’ is a binary feature for eating food that would be toggled on only if there are no ghosts nearby, and there’s food in our successor position. 
‘foodCarry’ is a continuous feature that would be toggled only if we’re carrying food. It takes the number of food we’re carrying and divides it by the distance to our base (calculated through BFS). This feature was really essential as it was put there to teach our agent whether wandering too far in the enemy’s base for food while having some amount of food to lose is a good or bad idea. 
‘depositFood’ is a binary feature that is toggled on only if the next state has a higher game score than ours. Essentially, it teaches our agent about depositing food. 
‘closestCapsule’ is a continuous feature that calculates the distance to the nearest power pellet that we can eat. This feature teaches our agent about minimizing how far capsules are from it at all times.
‘closest-food’ is a continuous feature that calculates the distance to the nearest food. This was the main feature that taught our agent about hunting food.
‘bias’ was set to one as it’s totally arbitrary anyway and our agent would learn it’s ‘true’ value over time

#### Reward Function: 
If we got closer to a food piece, the agent would be rewarded in an exponential function where the closer it is to a food piece the more reward it gets. However, if the agent moves away from a food piece, it gets penalized. This led to a somewhat greedy agent, which we wanted as we wanted to get food quickly, especially in the beginning.
If we take an action and die, the agent gets penalized significantly. This led to an agent that tries to live for as long as possible .
If our agent ate a scared ghost, it would get rewarded heavily. Oddly enough, our agent didn’t learn to eat scared ghosts or pursue them. We suspect this could be due to problems within our scared ghosts' features. 
If our agent ate food, we would give it a medium reward since there was a lot of food around, and we still didn’t want it to be super greedy.
If the agent increased our score, it’d get rewarded in relation to how much it increased our score plus a base reward.
If our agent ate a power pellet, it’d get rewarded meagerly since we didn’t want our agent to focus on that.

### Q Learning:
To learn, we would export two JSON files after the end of each episode with our weights and episodes data then import them once the game starts again into Counter objects.  We then set our agent to learn silently for 300 episodes with the following learning parameters: 
Alpha (α) was set to 0.75 as we wanted an agent that uses new information in training, but still retain a good portion of what it learned before as there were maps that had very unique features where we didn’t want our agent to retain a lot from.
Epsilon (ε) was set to 0.4. This was very important as Pacman is a very complicated game and there are a lot of multi-step actions that can lead to good positions, such as eating a close capsule then eating a ghost to escape your position or eating a Pacman when you were in defense. Thus, we needed an agent that explores a lot, especially in training time. We decided on 0.4 as we still didn’t want more random actions than logical actions.
Discount (𝛄) was set to 0.85 as we wanted our agent to not just use our reward as the only pointer to learn and not just the next state’s value. We wanted more emphasis on the value of the next state.
Results:
Approximate TD Q Learning resulted in almost our best agent so far. The agent would almost never get eaten, it beat all of our other agents in surviving by great margins. However, we would lose a few times to the baseline team due to our agent never learning to deposit food if we were far away from our base. This might be explainable that we didn’t have a feature that directly correlated with “you need to stop being greedy right now and deposit all your food!” “foodCarry” was put for that purpose, but it didn’t quite manage to teach our agent to deposit food. This could be explained by there not being a lot of series of random actions through epsilon that would teach this complex task to our agent. Our agent would go collect all the food most of the time and keep dancing in the enemy’s base. At that moment, we were close to the deadline and didn’t have lots of options, so we added code to force the agent to return to base after eating 25% of the food on the board. This code, however, wouldn’t take into consideration the enemy’s positions, which would result in our agent getting eaten a lot of the time on its way back. Furthermore, our agent never learned the complex task of eating a food capsule, then eating a ghost. This would lead to our agent still being scared of ghosts even when the enemy is scared. We assume this is most likely due to our scared ghost logic handling lacking depth to it. Since ghosts could still eat you when other ghosts are scared, this might’ve led to our agent still being scared of ghosts even after getting the power pellet. Nonetheless, this was by far our best agent as it would always beat the baseline team and almost always beat our other agents.

### Patches and improvements:
To get our offensive agent to work better, we thought of various ways to improve our agent through different strategies. Our agent had a habit of getting cornered easily due to it having just a depth of one as with Q Learning when choosing an action. This could be patched by arming our agent with minMax. We would use the state values we learned as the evaluation function of a minMax. This would allow us to look multiple steps ahead and find the best series of action within our depth. Furthermore, our agent would always pursue the closest food to it. This would lead sometimes to our agent pursuing a close food then a really far food after. This could be fixed through using a search algorithm, preferably A*, to search for the closest and least dangerous path to multiple foods. Our agent, further, doesn’t utilize the game-changing effect of power pellets. Better features that tie eating power pellets with eating ghosts with surviving will surely help. More importantly, teaching our agent to decrease greediness the more food it has to lose and run back to base or find a path that accumulates food as it runs to base would strengthen our offensive agent significantly. Less importantly, expanding our particle filtering algorithm is surely helpful as enemy agents can communicate with each other, but would need more computational time and, thus, time. Also, teaching our offensive agent to work in defense sometimes might be helpful as we only need 1 point to win not all possible points. Thus, an offensive agent that quickly snatches a food and then works in defense might lead to more, smaller wins. However, experimenting is needed with this first.
	

## Defense:
To create a defensive agent, we chose to start with a very basic defensive agent, and if we had time we’d improve it. This was a sound decision considering up to two days before the deadline, we had been working on an offensive agent without making so much improvement. Given this challenge, we came to the conclusion that a sub-optimal agent would probably do well considering how hard making an offensive agent was. We vowed we were gonna improve it, but we didn’t want perfection to be the enemy of good (we submitted the suboptimal agent, and did fairly well). With that said, we’ll dive into the strategies and techniques we used for our defensive agent.

### Strategies:
We thought that it would be best that our agent stays in our territory but in places where it believes that the enemy would be close to. Our agent naturally went towards the middle of the board at the beginning of the game, and proceeded to chase down the enemy once it believed an enemy breached our territory. Often, our agent did well enough to prevent the enemy from coming into our territory by forcing a stand-off no matter what the enemy tried to do. Better teams found ways to navigate this challenge by using A* star, or some other method to use longer paths into our territory, and our agent failed massively when this was done. Even then, many teams didn’t go this far—remember offensive agents were massively hard to build. Whenever an enemy ate the pellet and we were ghost, we thought it’d be best to waste the opponent’s moves on our board as we assumed once a pellet has been eaten, most people would want to eat the agent if it’s close. For this case, we’d still go near the opponent, but kind of danced around to waste their time and move. This strategy proved to be ridiculously effective–imagine you are pacman, you’re trying to collect food pellets, you see a scared ghost come close to you. As pacman, this could be dangerous as their scared timer could be well near over, so you try to eat the ghost and respawn them so you can go on with your business, but the ghost kind of just moves around you, but never too far from you! You end up using moves chasing the ghost around while it gains time on you and eventually when the scared timer goes off, it pounces!

### Techniques:
#### Particle filtering:
While our particle filtering wasn’t perfect, it worked well enough and we often tracked the enemy with high accuracy. The idea here is to see where in our territory was the probability of the enemy highest, then we used modified A* search to go in that direction. Generally, as we walked towards that direction, we got to within five distances of the enemy, so we could see them and hunt them directly. As we mentioned earlier, at the beginning of the game, it was often in the middle of the board since the opponent’s agents were trying to breach our territory. Many times, our agent was effective in the middle of the board, preventing the enemy’s agents from breaching our territory at all. Sometimes though, it failed as the opponents could use their own defensive agent as decoy and since our agent was focused on going to where there’s high probability of the enemy, we were stuck with the opponent’s defensive agent while their offense roamed our territory. Our agent would still hunt the offensive agent, but in this case it was often too late.
#### Alpha-beta search (minimax):
Since the game had pellets, that if an opponent eats, you become scared and edible for forty or so moves, this was a dangerous state to be in. We thought of preventing this by just defending pellets, but after running a couple of games with this strategy, we found it ineffective as we didn’t have enough agents to defend all pellets available in some games (one agent vs two pellets). We therefore let the enemy have the luxury of the pellets, but in that case we still got closer to the enemy, but with the goal of wasting their moves. Go near the opponent, but mini-max our way around them. We built an evaluation function to prioritize being away but close enough to our enemy. We used minimax when we were within two cell distances of the enemy and our scared timer was on. This often proved effective as we baited the enemy into eating us, but went around hence distracting them and once our timer was off, we pounced.
### Improvements:
Our defensive agent could obviously use a tonne of improvement. We’ve highlighted the obvious ones below:
We could have balanced our defensive agent to turn into an offensive agent especially when our scared timer would be on. This could’ve helped us avoid time and gain more on the offense.
We could have prioritized using the highest probability of the opponent in our territory rather than the highest probability of the opponent. This might have helped us filter out decoys and hunt ghosts more aggressively.
Implemented a learning defensive agent, and strategy so we don’t have to hard-code strategies. We believe that this could’ve helped mitigate the above (and other) challenges we faced.


## Final Results
During the class tournament, our agent came in 5th/41, winning 31 out of 40 games !!
### Demo
Below is a demo of our team (in red) playing a game in the tournament


https://user-images.githubusercontent.com/52356809/145605861-c4af8384-feb8-44cb-9175-caae42d86dd9.mov

