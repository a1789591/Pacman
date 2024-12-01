(define (domain ghost)
    (:requirements
        :typing
        :negative-preconditions
    )

    (:types
        enemyPacman cells
    )

    (:predicates
        (cell ?p)

        ;ghost's location
        (ghost-at ?loc - cells)

        ;enemy pacman location
        (enemy-at ?pac - enemyPacman ?loc - cells)

        ;capsule location
        (capsule-at ?loc - cells)

        ;connects cells
        (connected ?from ?to - cells)
    )
    
    (:action move
        :parameters (?from ?to - cells)
        :precondition (and 
                (ghost-at ?from)
                (connected ?from ?to)
                (not(enemy-at ?to))
        )
        :effect (and 
                (ghost-at ?to)
                (not(ghost-at ?from))
        )
    )

    (:action move-to-enemy
        :parameters (?from ?to - cells)
        :precondition (and
                    (ghost-at ?from)
                    (connected ?from ?to)
                    (enemy-at ?to)
        )
        :effect (and
                (ghost-at ?to)
                (not(ghost-at ?from))
                (not(enemy-at ?pac ?loc))
        )
    )
    

)