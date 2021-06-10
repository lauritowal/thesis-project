- prepare docker for titze: https://www.frontiersin.org/research-topics/19880/ai-in-aviation#articles
https://github.com/rtatze/JSBSim_gym_wrapper [IN PROGRESS]

- Increase km number to 3 instead of 2
- Instead of altitude ft to ground --> replace with rest altitude ft to runway... [DONE] 
- If on path --> reward for decreasing difference to target height (decreasing rest altitude) [DONE] 
- punish for rest distance instead of giving -10 on crash [DONE] 
- Try give more weight to vertical track error maybe?   
- Testing again in 3D --> Distance and heading error

- Train on elevevator instead of gamma hold [IN PROGRESS]
- Make the things you have now awesome, add some experiments with wind... and leave obstacles out... !

- Make plan for next steps

- Next: Wind
- Evaluate with wind
- Add wind into training ? 