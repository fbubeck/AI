$	 ????????LHP?????????!?٬?\m??$	X?]?@R?+???@?P?ɣ$	@!?}#?U?0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$[Ӽ????-???1??A??z6???Y]?C?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&(~??k	???Q???Aı.n???YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&n????h??s???AW?/?'??Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&`??"????!?lV}??A?[ A?c??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?*????[B>?٬??Az?):????Y?1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????&S???AS?!?uq??YvOjM??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?U??????.?!??u??AX?5?;N??Y?:pΈҞ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}??j?q?????A鷯????Y0L?
F%??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?٬?\m??HP?s??A?٬?\m??Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?Zd;???B?i?q??A4??7????Y-C??6??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
Q?|a???:M???A?~?:p???Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o??ʡ??3ı.n???A??ܵ??YjM??S??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q?????:M??Am???????Y??(????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$(~????6?>W[???A?ŏ1w??Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&4??7?????St$????A???N@??Y@?߾???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#??~j????|?5^???A???????Y??B?iޡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&k?w??#????H.???A?-???1??Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&aTR'?????|a2U??A?z?G???Y\ A?c̝?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>yX?5????? ?	??A????o??Y??镲??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e?c??????Aё\?C???Y??q????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????????????A??	h"??Y???Q???*	????̲?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat!?rh????!PdL2?A@)??T?????1r?.??@@:Preprocessing2F
Iterator::Model@?߾???!?N ?
A@)?>W[????1 ?· ?4@:Preprocessing2U
Iterator::Model::ParallelMapV2)?Ǻ???!B?????*@))?Ǻ???1B?????*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipaTR'????!??o??qP@)??e??a??1A?ł(?%@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????x???!???7?'3@)?JY?8ֵ?1`??w@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??Q???!n`?
-@)??Q???1n`?
-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate<Nё\???!?e?}??(@)Y?? ޲?1?jC?Bb@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?4?8EG??!??S?q?@)?4?8EG??1??S?q?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9E??î@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?AQr?@??'????????????!?:M???	!       "	!       *	!       2$	?Xe$>6??tx?k?????	h"??!?٬?\m??:	!       B	!       J$	???V_ȣ???$??S????q????!]?C?????R	!       Z$	???V_ȣ???$??S????q????!]?C?????JCPU_ONLYYE??î@b 