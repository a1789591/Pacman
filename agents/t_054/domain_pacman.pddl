(define (domain pacman)

    (:requirements
        :typing
        :negative-preconditions
    )

    (:types
        foods cells pacman ghost
    )

    (:predicates
        (cell ?p)
        ;get pacman location
        (pacman-at ?loc - cells)

        ;get food location
        (food-at ?loc - cells ?f - foods)

        ;get ghost location
        (ghost-at ?loc - cells)

        ;get capsule location
        (capsule-at ?loc - cells)

        ;connect two cells
        (connected ?from ?to - cells)

        ;food carried by pacman
        (food-carrying)

        ;have eaten a capsule - get the power to eat ghost, yay:)
        (has-super-power)

        ;eaten by ghost
        (eaten-by-ghost ?pacman)

        ;eaten by pacman
        (eaten-by-pacman ?ghost)

        ;wanna die to back home
        (die-to-home)
    )

    (:action move
        :parameters (?from ?to - cells)
        :precondition (and 
                (pacman-at ?from)
                (connected ?from ?to)
                (not(ghost-at ?to))
                (not(capsule-at ?to))
                (not(food-at ?to))
        )
        :effect (and 
                (pacman-at ?to)
                (not(pacman-at ?from))
        )
    )

    (:action move-to-capsule
        :parameters (?from ?to - cells)
        :precondition (and
                    (pacman-at ?from)
                    (connected ?from ?to)
                    (not(ghost-at ?to))
                    (not(food-at ?to))
                    (capsule-at ?to)
         )
        :effect (and 
                (pacman-at ?to)
                (not(pacman-at ?from))
                (has-super-power)
                (not(capsule-at ?to))
        )
    ) 

    (:action move-to-ghost
        :parameters (?from ?to - cells ?pacman pacman)
        :precondition (and 
                    (pacman-at ?from)
                    (connected ?from ?to)
                    (ghost-at ?to)
        )
        :effect (and 
                (pacman-at ?to)
                (not(pacman-at ?from))
                (eaten-by-ghost ?pacman)
        )
    )

    (:action move-to-ghost-invincible
        :parameters (?from ?to - cells ?ghost ghost)
        :precondition (and 
                    (pacman-at ?from)
                    (connected ?from ?to)
                    (ghost-at ?to)
                    (has-super-power)
        )
        :effect (and 
                (pacman-at ?to)
                (not(pacman-at ?from))
                (eaten-by-pacman ?ghost)
        )
    )

    
    (:action move-to-food
        :parameters (?from ?to - cells)
        :precondition (and 
                    (pacman-at ?from)
                    (connected ?from ?to)
                    (food-at ?to)
                    (not(ghost-at ?to))
                    (not(capsule-at ?to))
        )
        :effect (and 
                (pacman-at ?to)
                (not(pacman-at ?from))
                (food-carrying)
                (not(food-at ?to))
        )
    )

    (:action move-no-constraint
        :parameters (?from ?to - cells)
        :precondition (and 
                    (pacman-at ?from)
                    (connected ?from ?to)
                    (die-to-home)
        )
        :effect (and 
                (pacman-at ?to)
                (not(pacman-at ?from))
        )
    )
    

    
    
    
    







)