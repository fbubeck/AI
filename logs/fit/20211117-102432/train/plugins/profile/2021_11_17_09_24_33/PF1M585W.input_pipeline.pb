$	l>j????.?$/????ޓ??Z???!D????9??$	?ӏcB?@'yv2yM@LKb?˺	@!-`??l?2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ŏ1w-??Q?|a2??A?t?V??YHP?sײ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????x???h"lxz???AM??St$??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&9??m4????߾?3??A????????Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	h"lx???H?}8??A2U0*???Yp_?Q??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&_?L???M?O???A}гY????Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????????c?]K???AV-?????Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??镲???lV}???Aۊ?e????YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=?U???e?X???A?\?C????Y????????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?A?f????????K??A?s?????Y?(??0??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	???{?????C?l????Az6?>W[??Y?|a2U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?鷯????h o???A.???1???Ym???{???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&D????9??؁sF????Aa2U0*???Y??ݓ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}??b?????a??4??A?E??????Y&S????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y?):???3??7??A?`TR'???YbX9?Ȧ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?t????H?}8??A?e??a???Y?R?!?u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?f??j+??gDio????A??o_??Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&jM??St??2w-!???A?[ A?c??Y(~??k	??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&OjM??????V?/??AB>?٬???Y??6???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&a2U0*???Ǻ?????Ao??ʡ??Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?9#J{???n4??@???A?4?8EG??Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ޓ??Z???a??+e??A|??Pk???Y??j+????*	??????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatvOjM??!????qB@)?K7?A`??1?I??$A@:Preprocessing2F
Iterator::ModelKY?8????!?vikiA@)?W?2ı??1vik霼4@:Preprocessing2U
Iterator::Model::ParallelMapV2?ZB>????!q)?s,,@)?ZB>????1q)?s,,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??_vO??!t?DKJKP@)d;?O????1 ???$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceM?O????!??ze?(@)M?O????1??ze?(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenatejM??St??!Ǩ?2?+@)????S??1???o?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap&S????!??????1@)?b?=y??1??丈@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?:pΈҞ?!$??'??@)?:pΈҞ?1$??'??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?i???@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	~3?!???8?ю????a??+e??!؁sF????	!       "	!       *	!       2$	Vc??P\??ͅP?n{??|??Pk???!?[ A?c??:	!       B	!       J$	J?I??j????w??Q???z6?>??!HP?sײ?R	!       Z$	J?I??j????w??Q???z6?>??!HP?sײ?JCPU_ONLYY?i???@b 