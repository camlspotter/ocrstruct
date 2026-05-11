from pydantic import BaseModel
import re

from ocrstruct.middle import Middle, Block, block_title_level
from ocrstruct.middle_to_markdown import RenderOptions, block_to_markdown


class Block2(BaseModel):
    block : Block
    str : str
    pos : int


class Chunk(BaseModel):
    blocks : list[Block]
    str : str
    pos : int


class Chunked(BaseModel):
    without_overlap : list[Chunk]
    with_overlap : list[Chunk]


def chunk_middle(
    m : Middle, 
    o : RenderOptions, 
    nchars : int, 
    overlap_chars : int
) -> Chunked:
    totalchars = nchars + overlap_chars

    ss : list[str]
    bs : list[Block2]
    (ss, bs) = _chunk_middle_no_overlap(m, o, nchars)

    for s in ss:
        assert s

    sps : list[tuple[str, int]] = []
    pos = 0
    for s in ss:
        if s:
            sps.append((s, pos))
        pos += len(s)

    chunks : list[tuple[str,int]] = []
    for i, (s, pos) in enumerate(sps):
        buf = ''
        done = False
        for (s_, pos_) in sps[i+1:]:
            ls = s_.splitlines(keepends=True)
            for l in ls:
                # at least one line must be added
                if buf != '' and len(s) + len(buf) + len(l) > totalchars:
                    if s + buf:
                        chunks.append((s + buf, pos))
                    done = True
                    break
                buf += l
            if done:
                break
        if not done:
            if s + buf:
                chunks.append((s + buf, pos))

    all = ''.join(ss)
    for (s, pos) in chunks:
        assert all[pos:pos+len(s)] == s

    return Chunked(
        without_overlap= _bind_blocks(sps, bs), 
        with_overlap= _bind_blocks(chunks, bs)
    )


def _bind_blocks(cs : list[tuple[str,int]], bs : list[Block2]) -> list[Chunk]:
    bs_ = bs
    chunks : list[Chunk] = []
    for c in cs:
        (s, start) = c
        end = start + len(s) - 1

        # drop blocks left of c
        try:
            i = next((i for i, b in enumerate(bs_) if start <= b.pos))
        except StopIteration:
            print('cs', cs)
            print('bs', bs)
            print('c', c)
            print('bs_', bs_)
            assert False
        bs_ = bs_[i:]

        j = next((j for j, b in enumerate(bs_) if end < b.pos), len(bs_))
        matched = bs_[:j]

        matched2 = [b for b in bs if max(start, b.pos) < min(end, b.pos + len(b.str)) ]
        assert matched == matched2

        chunks.append(Chunk(
            blocks= [b.block for b in matched], 
            str= s, 
            pos= start,
        ))

    return chunks


def _chunk_middle_no_overlap(
    m : Middle, 
    o : RenderOptions, 
    nchars : int
) -> tuple[list[str], list[Block2]]:
    '''
    Split blocks w/o overlaps.

    Result: list of (chunk string, blocks)

    The same block may belong to more than 1 chunks, if its string is
    longer than `nchars`.
    '''
    bs_ : list[tuple[Block, str]] = [
        (b, _normalize_newlines(block_to_markdown(b, options=o)))
        for p in m.pdf_info for b in p.para_blocks
    ]
    # drop empty blocks
    bs_ = [b for b in bs_ if b[1].strip() != '']
    
    bs : list[Block2] = []
    pos = 0
    for (b, s) in bs_:
        bs.append(Block2(block=b, str=s, pos=pos))
        pos += len(s)

    bss = _split_blocks(bs, nchars)
    ss : list[str] = []
    for bs__ in bss:
        if len(bs_) == 1 and _blockstr_length(bs__) > nchars:
            ss.extend(_chunk_block(bs__[0], nchars))
        else:
            ss.append(''.join([b.str for b in bs__]))
    ss = [s for s in ss if s]

    # Check all the texts are in `ss`
    assert ''.join([b.str for b in bs]) == ''.join(ss)
    return (ss, bs)


def _normalize_newlines(s : str):
    return re.sub(r'\n{3,}$', '\n\n', s + '\n\n')


def _blockstr_length(bs : list[Block2]) -> int:
    return sum([len(b.str) for b in bs])


def _chunk_block(b : Block2, nchars : int) -> list[str]:
    if len(b.str) <= nchars:
        return [b.str]

    # For now only text blocks are chunked further
    if b.block.type != 'text':
        return [b.str]

    lines = b.str.splitlines(keepends=True)
    nchunks = int((len(b.str) + nchars - 1) / nchars)
    nchars2 = int(len(b.str) + nchunks - 1) / nchunks
    assert nchars2 <= nchars

    # We believe lines cannot be very very long since
    # they are printed on papers.

    chunks : list[str] = []
    buf = []
    buflen = 0
    for l in lines:
        if nchars2 <= buflen + len(l):
            if buf:
                s = ''.join(buf)
                chunks.append(s)
            buf = [l]
            buflen = len(l)
        else:
            buf.append(l)
            buflen += len(l)
    if buf:
        chunks.append(''.join(buf))

    assert b.str == ''.join(chunks)
    return chunks
    

def _check_block_grouping(bs : list[Block2], bss : list[list[Block2]]) -> None:
    assert [b for bs_ in bss for b in bs_] == bs


def _split_blocks(bs : list[Block2], nchars : int) -> list[list[Block2]]:
    bss = _group_blocks_by_titles_rec(bs, nchars, 1)
    
    bss2 = []
    for bs_ in bss:
        bss_ = _group_blocks(bs_, nchars)
        bss2.extend(bss_)
    _check_block_grouping(bs, bss2)
    return bss2


def _group_blocks(bs : list[Block2], nchars : int) -> list[list[Block2]]:
    if _blockstr_length(bs) <= nchars:
        return [bs]

    bss = []
    buf = []
    len = 0
    for b in bs:
        blen = _blockstr_length([b])
        if len + blen > nchars:
            if buf:
                bss.append(buf)
                buf = []
                len = 0
            buf = [b]
            len = blen
        else:
            len += blen
            buf.append(b)
    if buf:
        bss.append(buf)
    _check_block_grouping(bs, bss)
    return bss


def _group_blocks_by_titles_rec(bs : list[Block2], nchars : int, level : int) -> list[list[Block2]]:
    if _blockstr_length(bs) <= nchars:
        return [bs]

    bss = _group_blocks_by_titles(bs, level)
    if level >= 6:
        return bss
    
    bss2 : list[list[Block2]] = []
    buf : list[Block2] = []
    len = 0
    for bs in bss:
        bslen = _blockstr_length(bs)
        if bslen > nchars:
            if buf:
                bss2.append(buf)
            bss2.extend(_group_blocks_by_titles_rec(bs, nchars, level + 1))
            buf = []
            len = 0
        elif len + bslen > nchars:
            if buf:
                bss2.append(buf)
            buf = bs
            len = bslen
        else:
            buf.extend(bs)
            len += bslen
    if buf:
        bss2.append(buf)

    return bss2


def _group_blocks_by_titles(bs : list[Block2], level : int) -> list[list[Block2]]:
    g : list[list[Block2]] = []
    buf : list[Block2] = []
    for b in bs:
        if lev := block_title_level(b.block):
            if lev <= level:
                if buf:
                    g.append(buf)
                buf = [b]
            else:
                buf.append(b)
        else:
            buf.append(b)
    if buf:
        g.append(buf)
    _check_block_grouping(bs, g)
    return g
