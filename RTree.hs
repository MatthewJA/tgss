module RTree where

import Data.List (sort)
import Data.Ord (comparing)

data Box = Box {topLeft :: Coord, bottomRight :: Coord}
    -- topLeft < bottomRight
    deriving Show

data RTree a = Empty | Leaf Box a | Node Box [RTree a]  -- length [RTree] <= 4

type Coord = (Float, Float)

instance Show a => Show (RTree a) where
    show tree = case tree of
        Empty -> ""
        Leaf box a -> show a ++ " (" ++ show box ++ ") "
        Node box children -> "Node (" ++ show box ++ ")\n" ++
            (concat
            . map (unlines . map ("\t"++) . lines)
            . map show
            $ children)

box :: RTree a -> Box
box tree = case tree of
    Leaf box _ -> box
    Node box _ -> box

search :: RTree a -> Coord -> [a]
search tree coord = case tree of
    Leaf _ a -> [a]
    Node _ children -> concatMap (flip search coord) $
                       filter (inBox coord . box) children

fromList :: [(a, Box)] -> RTree a
fromList [] = Empty
fromList [(a, box)] = Leaf box a
fromList list = Node (mergeBoxList $ map box childrenTrees) childrenTrees
    where
        childrenTrees = filter notEmpty . map fromList $ partitions list

        mergeBoxList :: [Box] -> Box
        mergeBoxList [] = error "Merging empty box list."
        mergeBoxList (x:xs) = foldr mergeBoxes x xs

        mergeBoxes :: Box -> Box -> Box
        mergeBoxes boxA boxB = Box
            (min (fst $ topLeft boxA) (fst $ topLeft boxB),
             min (snd $ topLeft boxA) (snd $ topLeft boxB))
            (max (fst $ bottomRight boxA) (fst $ bottomRight boxB),
             max (snd $ bottomRight boxA) (snd $ bottomRight boxB))

        notEmpty :: RTree a -> Bool
        notEmpty Empty = False
        notEmpty _ = True

        partitions :: [(a, Box)] -> [[(a, Box)]]
        partitions list = do
            xOp <- [(>=medianX), (<medianX)]
            yOp <- [(>=medianY), (<medianY)]
            return . filterBy fst xOp . filterBy snd yOp $ list

        filterBy getter pred = filter (pred . boxMiddle getter . snd)

        median :: (Coord -> Float) -> [(a, Box)] -> Float
        median getter list
            | null list = error "Median of empty list."
            | odd $ length list = sorted !! midway
            | otherwise = (
                (sorted !! (midway - 1)) +
                (sorted !! midway)) / 2
            where sorted = sort (map (boxMiddle getter . snd) list)
        medianX = median fst list
        medianY = median snd list

        midway = length list `div` 2

        boxMiddle getter box = (getter (topLeft box) + getter (bottomRight box)) / 2

inBox :: Coord -> Box -> Bool
inBox (a, d) box = a >= x && a < x + w && d >= y && d < y + h
    where
        (x, y) = topLeft box
        (x2, y2) = bottomRight box
        (w, h) = (x2 - x, y2 - y)

main :: IO ()
main = undefined
